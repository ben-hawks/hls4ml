from pathlib import Path

import numpy as np
import pytest

from hls4ml.backends.fpga.fpga_backend import FPGABackend
from hls4ml.model.types import CodebookPrecisionType

test_root_path = Path(__file__).parent


@pytest.mark.parametrize('precision', ['codebook<4,12>', 'codebook< 8 , 16 >'])
def test_codebook_parsing(precision):
    ctype = FPGABackend.convert_precision_string(precision)
    assert isinstance(ctype, CodebookPrecisionType)
    assert FPGABackend.convert_precision_string(str(ctype)) == ctype
    assert CodebookPrecisionType.deserialize(ctype.serialize_state()) == ctype


@pytest.mark.parametrize('precision', ['codebook<4>', 'codebook<4,12,1>', 'codebook<a,b>'])
def test_codebook_parsing_invalid(precision):
    with pytest.raises(ValueError):
        FPGABackend.convert_precision_string(precision)


@pytest.mark.parametrize('width, decode_width', [(2, 8), (4, 12), (6, 12), (8, 16)])
@pytest.mark.parametrize('scale', [0.05, 0.5, 4.0])
def test_codebook_fitted(width, decode_width, scale):
    data = np.random.default_rng(4).normal(scale=scale, size=(32, 16))
    ctype = CodebookPrecisionType(width, decode_width)
    with pytest.raises(RuntimeError):
        ctype.encode(data)

    fit = ctype.fitted(data)
    assert fit.is_fitted and fit.uses_table
    assert CodebookPrecisionType.deserialize(fit.serialize_state()) == fit
    values = np.array(fit.values)
    assert 1 < len(values) <= 1 << width
    assert np.all(np.diff(values) > 0)

    # Entries are representable in the decode type, which has no integer bits to spare
    dp = fit.decode_precision()
    assert dp.width == decode_width
    assert np.all(np.mod(values * 2.0**dp.fractional, 1) == 0)
    assert 2.0 ** (dp.integer - 2) <= np.max(np.abs(values)) < 2.0 ** (dp.integer - 1)

    # Every value maps to its nearest entry
    codes = fit.encode(data)
    assert codes.min() >= 0 and codes.max() < len(values)
    dist = np.abs(data[..., None] - values).min(axis=-1)
    np.testing.assert_allclose(np.abs(data - fit.quantize(data)), dist, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(fit.lookup_table()[codes], fit.quantize(data))

    # Refitting to already quantized data reproduces the same table
    assert ctype.fitted(fit.quantize(data)) == fit


def _brute_force_kmeans(x, k):
    import itertools

    x = np.sort(x)
    best = np.inf
    for cuts in itertools.combinations(range(1, len(x)), k - 1):
        bounds = (0,) + cuts + (len(x),)
        best = min(best, sum(((x[a:b] - x[a:b].mean()) ** 2).sum() for a, b in zip(bounds[:-1], bounds[1:])))
    return best


@pytest.mark.parametrize('seed', range(20))
def test_codebook_kmeans_optimal(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 10))
    k = int(rng.integers(2, n))
    x = rng.laplace(size=n)
    centers = CodebookPrecisionType._kmeans_1d(x, k)
    idx = np.searchsorted((centers[1:] + centers[:-1]) / 2, np.sort(x))
    np.testing.assert_allclose(((np.sort(x) - centers[idx]) ** 2).sum(), _brute_force_kmeans(x, k))


def _quantize_weights(model, ctype):
    """Replace the kernels by the values ``ctype`` stores, so the codebook and fixed point models see the same numbers."""
    for layer in model.layers:
        if not layer.get_weights():
            continue
        kernel, bias = layer.get_weights()
        layer.set_weights([ctype.fitted(kernel).quantize(kernel).astype(np.float32), bias])
    return model


def _check_codebook_weights(test_case_id, model, ctype, layers, X, io_type, strategy, atol):
    import hls4ml

    predictions = {}
    weight_precisions = dict.fromkeys(layers, str(ctype))
    for suffix in ('codebook', 'fixed'):
        config = hls4ml.utils.config_from_keras_model(
            model, granularity='name', default_precision='fixed<16,6>', backend='Vitis'
        )
        config['Model']['Strategy'] = strategy
        for name in layers:
            config['LayerName'][name]['Precision']['weight'] = weight_precisions[name]
            config['LayerName'][name]['ReuseFactor'] = 4 if strategy == 'Resource' else 1

        output_dir = str(test_root_path / f'{test_case_id}_{suffix}')
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, output_dir=output_dir, backend='Vitis', io_type=io_type
        )
        hls_model.compile()
        predictions[suffix] = hls_model.predict(X)

        if suffix == 'codebook':
            defines = Path(output_dir, 'firmware', 'defines.h').read_text()
            for name in layers:
                # The fixed point reference stores the same values directly, in the decode type
                wp = hls_model.graph[name].get_attr('weight_t').precision
                assert wp.is_fitted
                dp = wp.decode_precision()
                lut = f'nnet::lut_weight<{ctype.width}, ap_fixed<{dp.width},{dp.integer}>, {name}_weight_t_table>'
                assert lut in defines
                weight_precisions[name] = f'fixed<{dp.width},{dp.integer}>'

    # Decoding the stored codes must give results bit-identical to storing the decoded values
    np.testing.assert_array_equal(predictions['codebook'], predictions['fixed'])
    np.testing.assert_allclose(
        predictions['codebook'].reshape(len(X), -1), model.predict(X, verbose=0).reshape(len(X), -1), rtol=0, atol=atol
    )


IO_STRATEGIES = [(io, strategy) for io in ('io_parallel', 'io_stream') for strategy in ('Latency', 'Resource')]


@pytest.mark.parametrize('width, decode_width', [(4, 12), (8, 16)])
@pytest.mark.parametrize('io_type, strategy', IO_STRATEGIES)
def test_codebook_weights_dense(test_case_id, width, decode_width, io_type, strategy):
    import keras

    keras.utils.set_random_seed(0)
    model = keras.Sequential(
        [
            keras.layers.Input((16,)),
            keras.layers.Dense(12, activation='relu', name='dense1'),
            keras.layers.Dense(5, name='dense2'),
        ]
    )
    ctype = CodebookPrecisionType(width, decode_width)
    X = np.random.default_rng(1).uniform(-2, 2, size=(200, 16)).astype(np.float32)
    _check_codebook_weights(
        test_case_id, _quantize_weights(model, ctype), ctype, ('dense1', 'dense2'), X, io_type, strategy, 5e-2
    )


@pytest.mark.parametrize('dim', [1, 2])
@pytest.mark.parametrize('io_type, strategy', IO_STRATEGIES)
def test_codebook_weights_conv(test_case_id, dim, io_type, strategy):
    import keras

    conv = keras.layers.Conv1D if dim == 1 else keras.layers.Conv2D
    keras.utils.set_random_seed(0)
    model = keras.Sequential(
        [
            keras.layers.Input((8,) * dim + (3,)),
            conv(6, (3,) * dim, activation='relu', name='conv1'),
            conv(4, (3,) * dim, name='conv2'),
        ]
    )
    ctype = CodebookPrecisionType(4, 12)
    X = np.random.default_rng(1).uniform(-1, 1, size=(50,) + (8,) * dim + (3,)).astype(np.float32)
    _check_codebook_weights(
        test_case_id, _quantize_weights(model, ctype), ctype, ('conv1', 'conv2'), X, io_type, strategy, 5e-2
    )
