from pathlib import Path

import numpy as np
import pytest

from hls4ml.backends.fpga.fpga_backend import FPGABackend
from hls4ml.model.types import CodebookPrecisionType, FixedPrecisionType, PositPrecisionType

test_root_path = Path(__file__).parent


@pytest.mark.parametrize('precision', ['posit<8,0>', 'posit<8, 1>', 'posit< 16 ,2 >'])
def test_posit_parsing(precision):
    ptype = FPGABackend.convert_precision_string(precision)
    assert isinstance(ptype, PositPrecisionType)
    assert FPGABackend.convert_precision_string(str(ptype)) == ptype
    assert PositPrecisionType.deserialize(ptype.serialize_state()) == ptype


@pytest.mark.parametrize('precision', ['posit<8>', 'posit<8,0,1>', 'posit<a,b>'])
def test_posit_parsing_invalid(precision):
    with pytest.raises(ValueError):
        FPGABackend.convert_precision_string(precision)


def test_posit_known_values():
    p8 = PositPrecisionType(8, 0)
    np.testing.assert_array_equal(p8.decode([0x00, 0x40, 0x60, 0x20, 0x7F, 0x01, 0xC0]), [0, 1, 2, 0.5, 64, 1 / 64, -1])
    assert np.isnan(p8.decode([0x80])[0])
    assert p8.decode_precision() == FixedPrecisionType(14, 8, True)

    p16 = PositPrecisionType(16, 1)
    np.testing.assert_array_equal(p16.decode([0x4000, 0x5000, 0x3000, 0x7FFF]), [1, 2, 0.5, 2.0**28])


@pytest.mark.parametrize('width, es', [(4, 0), (6, 1), (8, 0), (8, 1), (8, 2), (10, 1)])
def test_posit_encode_decode(width, es):
    ptype = PositPrecisionType(width, es)
    codes = np.arange(1 << width)
    codes = codes[codes != 1 << (width - 1)]  # NaR
    values = ptype.decode(codes)

    # Every value is exactly representable by the decode type
    dp = ptype.decode_precision()
    assert np.all(np.mod(values * 2.0**dp.fractional, 1) == 0)
    assert np.max(np.abs(values)) < 2.0 ** (dp.integer - 1)

    # Encoding is the inverse of decoding
    np.testing.assert_array_equal(ptype.encode(values), codes)

    # Rounding picks the nearest value, saturates and never rounds a non-zero value to zero
    rng = np.random.default_rng(0)
    x = rng.normal(scale=2.0, size=1000)
    q = ptype.quantize(x)
    nonzero = np.sort(np.abs(values[values != 0]))
    dist = np.abs(x[:, None] - values[None, values != 0]).min(axis=1)
    np.testing.assert_allclose(np.abs(x - q), dist, rtol=0, atol=0)
    assert ptype.quantize([1e30])[0] == nonzero[-1]
    assert ptype.quantize([-1e30])[0] == -nonzero[-1]
    assert ptype.quantize([1e-30])[0] == nonzero[0]
    assert ptype.quantize([0.0])[0] == 0


@pytest.mark.parametrize('width, es', [(6, 1), (8, 0), (8, 1), (8, 2)])
@pytest.mark.parametrize('scale', [0.05, 0.5, 4.0])
def test_posit_fitted(width, es, scale):
    ptype = PositPrecisionType(width, es)
    data = np.random.default_rng(2).normal(scale=scale, size=(16, 12))
    fit = ptype.fitted(data)
    assert fit.is_fitted and not ptype.is_fitted
    assert fit.uses_table
    assert PositPrecisionType.deserialize(fit.serialize_state()) == fit
    assert np.array_equal(fit.encode(data), ptype.encode(data))

    # The table holds the decoded value of every code the data uses, and nothing else
    codes = ptype.encode(data)
    table = fit.lookup_table()
    assert set(fit.used_codes) == set(np.unique(codes))
    np.testing.assert_array_equal(table[codes], ptype.quantize(data))
    unused = np.setdiff1d(np.arange(1 << width), fit.used_codes)
    assert np.all(table[unused] == 0)

    def exact(w, i):
        q = ptype.quantize(data)
        return np.all(np.mod(q * 2.0 ** (w - i), 1) == 0) and np.all(np.abs(q) < 2.0 ** (i - 1))

    dp = fit.decode_precision()
    full = ptype.decode_precision()
    assert dp.width <= full.width and dp.integer <= full.integer
    assert exact(dp.width, dp.integer)
    # One bit fewer integer or fractional bits is no longer exact
    assert not exact(dp.width - 1, dp.integer - 1)
    assert not exact(dp.width - 1, dp.integer)


def test_posit_wide_has_no_table():
    fit = PositPrecisionType(12, 1).fitted(np.random.default_rng(3).normal(size=100))
    assert fit.is_fitted and not fit.uses_table


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


def _model(seed=0):
    import keras

    keras.utils.set_random_seed(seed)
    return keras.Sequential(
        [
            keras.layers.Input((16,)),
            keras.layers.Dense(12, activation='relu', name='dense1'),
            keras.layers.Dense(5, name='dense2'),
        ]
    )


def _quantize_weights(model, ptype):
    """Replace the kernels by the values ``ptype`` stores, so the table and fixed point models see the same numbers."""
    for layer in model.layers:
        kernel, bias = layer.get_weights()
        layer.set_weights([ptype.fitted(kernel).quantize(kernel).astype(np.float32), bias])
    return model


def _check_table_weights(test_case_id, model, ptype, io_type, strategy):
    import hls4ml

    X = np.random.default_rng(1).uniform(-2, 2, size=(200, 16)).astype(np.float32)
    predictions = {}
    weight_precisions = {'dense1': str(ptype), 'dense2': str(ptype)}
    for suffix in ('table', 'fixed'):
        config = hls4ml.utils.config_from_keras_model(
            model, granularity='name', default_precision='fixed<16,6>', backend='Vitis'
        )
        config['Model']['Strategy'] = strategy
        for name in ('dense1', 'dense2'):
            config['LayerName'][name]['Precision']['weight'] = weight_precisions[name]
            config['LayerName'][name]['ReuseFactor'] = 4 if strategy == 'Resource' else 1

        output_dir = str(test_root_path / f'{test_case_id}_{suffix}')
        hls_model = hls4ml.converters.convert_from_keras_model(
            model, hls_config=config, output_dir=output_dir, backend='Vitis', io_type=io_type
        )
        hls_model.compile()
        predictions[suffix] = hls_model.predict(X)

        if suffix == 'table':
            defines = Path(output_dir, 'firmware', 'defines.h').read_text()
            for name in ('dense1', 'dense2'):
                # The fixed point reference stores the same values directly, in the decode type
                wp = hls_model.graph[name].get_attr('weight_t').precision
                assert wp.is_fitted
                dp = wp.decode_precision()
                if wp.uses_table:
                    lut = f'nnet::lut_weight<{ptype.width}, ap_fixed<{dp.width},{dp.integer}>, {name}_weight_t_table>'
                    assert lut in defines
                else:
                    assert f'nnet::posit<{ptype.width},{ptype.es},{dp.width},{dp.integer}>' in defines
                weight_precisions[name] = f'fixed<{dp.width},{dp.integer}>'
            weights = Path(output_dir, 'firmware', 'weights', 'w2.h').read_text()
            codes = weights.split('= {')[1].split('}')[0].split(',')
            assert all(0 <= int(c) < (1 << ptype.width) for c in codes)

    # Decoding the stored codes must give results bit-identical to storing the decoded values
    np.testing.assert_array_equal(predictions['table'], predictions['fixed'])
    np.testing.assert_allclose(predictions['table'], model.predict(X, verbose=0), rtol=0, atol=5e-2)


IO_STRATEGIES = [
    ('io_parallel', 'Latency'),
    ('io_parallel', 'Resource'),
    ('io_stream', 'Latency'),
    ('io_stream', 'Resource'),
]


@pytest.mark.parametrize('width, es', [(8, 0), (6, 1), (12, 1)])
@pytest.mark.parametrize('io_type, strategy', IO_STRATEGIES)
def test_posit_weights_dense(test_case_id, width, es, io_type, strategy):
    ptype = PositPrecisionType(width, es)
    _check_table_weights(test_case_id, _quantize_weights(_model(), ptype), ptype, io_type, strategy)


@pytest.mark.parametrize('width, decode_width', [(4, 12), (8, 16)])
@pytest.mark.parametrize('io_type, strategy', IO_STRATEGIES)
def test_codebook_weights_dense(test_case_id, width, decode_width, io_type, strategy):
    ctype = CodebookPrecisionType(width, decode_width)
    _check_table_weights(test_case_id, _quantize_weights(_model(), ctype), ctype, io_type, strategy)
