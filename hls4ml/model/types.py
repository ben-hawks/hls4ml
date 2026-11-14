"""
This module contains the definitions of classes hls4ml uses to represent data types. The data types are equivalents of
C++/HLS data types. The basic type(``PrecisionType``) is defined as having a specified width in bits (it's 'precision').
The Precision types are given names for convenience (``NamedType``). Named types are the building blocks of
higher-dimensional tensors, which are defined as arrays or FIFO streams in the generated code.
"""

import copy
import math
from enum import Enum

import numpy as np

# region Serialization


class Serializable:
    """Classes should implement this interface to provide serialization support.

    Objects are serialized into a JSON format, with two fields, ``class_name`` and ``state``. Objects need to provide both.
    ``class_name`` is used to map the object to the class which which can deserialize the object via ``deserialize`` class
    method. ``state`` represents the current state of the object and will be passed to ``deserialize``. Implementations are
    expected to capture all internal state to be able to recreate the object indistinguishable from the original.
    """

    def serialize_class_name(self):
        # Wrapped classes are serialized as original types, since many of wrapped classes are created dynamically.
        if hasattr(self, '_wrapped'):
            cls_name = self._wrapped
        else:
            cls = self.__class__
            cls_name = cls.__module__ + '.' + cls.__qualname__

        return cls_name

    def serialize_state(self):
        raise NotImplementedError

    def serialize(self):
        return {'class_name': self.serialize_class_name(), 'state': self.serialize_state()}

    @classmethod
    def deserialize(cls, state):
        return cls(**state)


# endregion

# region Precision types


class RoundingMode(Enum):
    TRN = 1
    TRN_ZERO = 2
    RND = 3
    RND_ZERO = 4
    RND_INF = 5
    RND_MIN_INF = 6
    RND_CONV = 7

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, mode):
        mode = mode.strip().replace('AP_', '').upper()
        mode = mode.strip().replace('AC_', '').upper()

        return cls[mode]


class SaturationMode(Enum):
    WRAP = 1
    SAT = 2
    SAT_ZERO = 3
    SAT_SYM = 4

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, mode):
        mode = mode.strip().replace('AP_', '').upper()
        mode = mode.strip().replace('AC_', '').upper()

        return cls[mode]


class PrecisionType(Serializable):
    """
    Base class representing a precision type of specified width.

    Subclasses of this provide concrete implementations of arbitrary precision integer and fixed-point types.

    Args:
        width (int): Number of bits used by the precision type.
        signed (bool): Signed or unsigned type.
    """

    def __init__(self, width, signed):
        self.width = width
        self.signed = signed

    def __eq__(self, other: object) -> bool:
        eq = self.width == other.width
        eq = eq and self.signed == other.signed

        return eq

    def __hash__(self) -> int:
        return hash((self.width, self.signed))

    def serialize_state(self):
        state = {
            'width': self.width,
            'signed': self.signed,
        }
        return state


class IntegerPrecisionType(PrecisionType):
    """Arbitrary precision integer  data type.

    This type is equivalent to ap_(u)int and ac_int HLS types.

    Args:
        width (int, optional): Number of bits used. Defaults to 16.
        signed (bool, optional): Signed or unsigned type. Defaults to ``True``.
    """

    def __init__(self, width=16, signed=True):
        super().__init__(width=width, signed=signed)

    def __str__(self):
        typestring = '{signed}int<{width}>'.format(signed='u' if not self.signed else '', width=self.width)
        return typestring

    # Does this need to make sure other is also an IntegerPrecisionType? I could see a match between Fixed and Integer
    def __eq__(self, other: object) -> bool:
        if isinstance(other, IntegerPrecisionType):
            return super().__eq__(other)

        return False

    def __hash__(self) -> int:
        return super().__hash__()

    @property
    def integer(self):
        return self.width

    @property
    def fractional(self):
        return 0

    @property
    def rounding_mode(self):
        return RoundingMode.TRN

    @property
    def saturation_mode(self):
        return SaturationMode.WRAP

    @property
    def saturation_bits(self):
        return 0


class FixedPrecisionType(PrecisionType):
    """Arbitrary precision fixed-point data type.

    This type is equivalent to ap_(u)fixed and ac_fixed HLS types.

    Args:
        width (int, optional): Total number of bits used. Defaults to 16.
        integer (int, optional): Number of integer bits left of the decimal point. Defaults to 6.
        signed (bool, optional): Signed or unsigned type. Defaults to ``True``.
        rounding_mode (RoundingMode, optional): Quantization mode. Defaults to ``None`` (TRN).
        saturation_mode (SaturationMode, optional): Overflow mode. Defaults to ``None`` (WRAP).
        saturation_bits (int, optional): The number of saturation bits. Defaults to ``None``.
    """

    def __init__(self, width=16, integer=6, signed=True, rounding_mode=None, saturation_mode=None, saturation_bits=None):
        super().__init__(width=width, signed=signed)
        self.integer = integer
        self.rounding_mode = rounding_mode
        self.saturation_mode = saturation_mode
        self.saturation_bits = saturation_bits

    # make this a property to avoid inconsistencies

    @property
    def fractional(self):
        return self.width - self.integer

    @property
    def rounding_mode(self):
        return self._rounding_mode

    @rounding_mode.setter
    def rounding_mode(self, mode):
        if mode is None:
            self._rounding_mode = RoundingMode.TRN
        elif isinstance(mode, str):
            self._rounding_mode = RoundingMode.from_string(mode)
        else:
            self._rounding_mode = mode

    @property
    def saturation_mode(self):
        return self._saturation_mode

    @saturation_mode.setter
    def saturation_mode(self, mode):
        if mode is None:
            self._saturation_mode = SaturationMode.WRAP
        elif isinstance(mode, str):
            self._saturation_mode = SaturationMode.from_string(mode)
        else:
            self._saturation_mode = mode

    @property
    def saturation_bits(self):
        return self._saturation_bits

    @saturation_bits.setter
    def saturation_bits(self, bits):
        if bits is None:
            self._saturation_bits = 0
        else:
            self._saturation_bits = bits

    def __str__(self):
        args = [self.width, self.integer, self.rounding_mode, self.saturation_mode, self.saturation_bits]
        args = ','.join([str(arg) for arg in args])
        typestring = '{signed}fixed<{args}>'.format(signed='u' if not self.signed else '', args=args)
        return typestring

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FixedPrecisionType):
            eq = super().__eq__(other)
            eq = eq and self.integer == other.integer
            eq = eq and self.rounding_mode == other.rounding_mode
            eq = eq and self.saturation_mode == other.saturation_mode
            eq = eq and self.saturation_bits == other.saturation_bits
            return eq

        return False

    def __hash__(self) -> int:
        return super().__hash__() ^ hash((self.integer, self.rounding_mode, self.saturation_mode, self.saturation_bits))

    def serialize_state(self):
        state = super().serialize_state()
        state.update(
            {
                'integer': self.integer,
                'rounding_mode': str(self.rounding_mode),
                'saturation_mode': str(self.saturation_mode),
                'saturation_bits': self.saturation_bits,
            }
        )
        return state

    @property
    def min(self):
        if not self.signed:
            return 0.0
        if self.saturation_mode == SaturationMode.SAT_SYM:
            return -(2.0 ** (self.integer - 1)) + 2.0**-self.fractional
        return -(2.0 ** (self.integer - 1))

    @property
    def max(self):
        return 2.0 ** (self.integer - self.signed) - 2.0**-self.fractional


class XnorPrecisionType(PrecisionType):
    """
    Convenience class to differentiate 'regular' integers from BNN Xnor ones
    """

    def __init__(self):
        super().__init__(width=1, signed=False)
        self.integer = 1

    # TODO:  this should really be a specific type
    def __str__(self):
        typestring = 'uint<1>'
        return typestring


class ExponentPrecisionType(PrecisionType):
    """
    Convenience class to differentiate 'regular' integers from those which represent exponents,
    for QKeras po2 quantizers, for example.
    """

    def __init__(self, width=16, signed=True):
        super().__init__(width=width, signed=signed)

    # TODO:  this should really be a specific type, not int
    def __str__(self):
        typestring = '{signed}int<{width}>'.format(signed='u' if not self.signed else '', width=self.width)
        return typestring


class FloatPrecisionType(PrecisionType):
    """
    Class representing a floating-point precision type.

    This type is equivalent to ac_float HLS types. If the use of C++ equivalent types is required, see
    ``StandardFloatPrecisionType``.

    Args:
        width (int, optional): Total number of bits used. Defaults to 33.
        integer (int, optional): Number of bits used for the integer part. Defaults to 2.
        exponent (int, optional): Number of bits used for the exponent. Defaults to 8.
    """

    def __init__(self, width=33, integer=2, exponent=8, rounding_mode=None):
        super().__init__(width=width, signed=True)
        self.exponent = exponent
        self.integer = integer  # If None, will be set to width - exponent - 1
        self.rounding_mode = rounding_mode

    @property
    def rounding_mode(self):
        return self._rounding_mode

    @rounding_mode.setter
    def rounding_mode(self, mode):
        if mode is None:
            self._rounding_mode = RoundingMode.TRN
        elif isinstance(mode, str):
            self._rounding_mode = RoundingMode.from_string(mode)
        else:
            self._rounding_mode = mode

    def __str__(self):
        args = [self.width - self.exponent, self.integer, self.exponent, self.rounding_mode]
        args = ','.join([str(arg) for arg in args])
        typestring = f'float<{args}>'
        return typestring

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FloatPrecisionType):
            eq = super().__eq__(other)
            eq = eq and self.integer == other.integer
            eq = eq and self.exponent == other.exponent
            eq = eq and self.rounding_mode == other.rounding_mode
            return eq

        return False

    def __hash__(self) -> int:
        return super().__hash__() ^ hash((self.integer, self.exponent, self.rounding_mode))

    def serialize_state(self):
        state = super().serialize_state()
        state.update(
            {
                'integer': self.integer,
                'exponent': self.exponent,
                'rounding_mode': str(self.rounding_mode),
            }
        )
        return state


class StandardFloatPrecisionType(PrecisionType):
    """
    Class representing a floating-point precision type.

    This type is equivalent to ap_float and ac_std_float HLS types. <32,8> corresponds to a 'float' type in C/C++. <64,11>
    corresponds to a 'double' type in C/C++. <16,5> corresponds to a 'half' type in C/C++. <16,8> corresponds to a
    'bfloat16' type in C/C++.

    Args:
        width (int, optional): Total number of bits used. Defaults to 32.
        exponent (int, optional): Number of bits used for the exponent. Defaults to 8.
        use_cpp_type (bool, optional): Use C++ equivalent types if available. Defaults to ``True``.
    """

    def __init__(self, width=32, exponent=8, use_cpp_type=True):
        super().__init__(width=width, signed=True)
        self.exponent = exponent
        self.use_cpp_type = use_cpp_type

    def __str__(self):
        if self._check_cpp_type(32, 8):
            typestring = 'float'
        elif self._check_cpp_type(64, 11):
            typestring = 'double'
        elif self._check_cpp_type(16, 5):
            typestring = 'half'
        elif self._check_cpp_type(16, 8):
            typestring = 'bfloat16'
        else:
            typestring = f'std_float<{self.width},{self.exponent}>'
        return typestring

    def _check_cpp_type(self, width, exponent):
        return self.use_cpp_type and self.width == width and self.exponent == exponent

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FloatPrecisionType):
            eq = super().__eq__(other)
            eq = eq and self.exponent == other.exponent
            return eq

        return False

    def __hash__(self) -> int:
        return super().__hash__() ^ hash(self.exponent)

    def serialize_state(self):
        state = super().serialize_state()
        state.update(
            {
                'exponent': self.exponent,
                'use_cpp_type': self.use_cpp_type,
            }
        )
        return state


class CodebookPrecisionType(PrecisionType):
    """
    Class representing a codebook<width, decode_width> type, used for weight storage only.

    Values are stored as ``width``-bit indices into a table of up to ``2**width`` signed fixed-point values with
    ``decode_width`` bits. ``fitted()`` chooses the table for a given tensor with optimal 1-D k-means, then rounds the
    entries to a fixed-point type whose integer bits cover the tensor's range. Arithmetic is done in fixed point.

    Args:
        width (int, optional): Number of index bits. Defaults to 4.
        decode_width (int, optional): Width of the fixed-point table entries. Defaults to 16.
        decode_integer (int, optional): Integer bits of the table entries. Defaults to ``None`` (set by ``fitted()``).
        values (list, optional): Table entries in increasing order. Defaults to ``None`` (set by ``fitted()``).
    """

    def __init__(self, width=4, decode_width=16, decode_integer=None, values=None):
        if width < 1:
            raise ValueError(f'Codebook width must be at least 1, got {width}')
        if decode_width < 2:
            raise ValueError(f'Codebook decode_width must be at least 2, got {decode_width}')
        super().__init__(width=width, signed=True)
        self.decode_width = decode_width
        self.decode_integer = decode_integer
        self.values = None if values is None else tuple(float(v) for v in values)

    def __str__(self):
        return f'codebook<{self.width},{self.decode_width}>'

    def _key(self):
        return (self.width, self.decode_width, self.decode_integer, self.values)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CodebookPrecisionType) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    @property
    def is_fitted(self):
        return self.values is not None

    @property
    def uses_table(self):
        return self.is_fitted

    def decode_precision(self):
        """The fixed-point type of the table entries."""
        if not self.is_fitted:
            raise RuntimeError(f'{self} has not been fitted to any data')
        return FixedPrecisionType(width=self.decode_width, integer=self.decode_integer, signed=True)

    max_bins = 4096  # Larger tensors are binned before clustering, which bounds the cost of the exact fit

    @classmethod
    def _kmeans_1d(cls, x, k):
        """
        Optimal 1-D k-means (minimum squared error), by dynamic programming over the sorted values with the
        divide-and-conquer speed-up that the monotone optimal split points allow. Values are first merged into at most
        ``max_bins`` equal-width bins, each standing at the mean of its values.
        """
        x = np.sort(np.asarray(x, dtype=np.float64).ravel())
        points, counts = np.unique(x, return_counts=True)
        if len(points) > cls.max_bins:
            edges = np.linspace(points[0], points[-1], cls.max_bins + 1)[1:-1]
            idx = np.searchsorted(edges, x, side='right')
            counts = np.bincount(idx, minlength=cls.max_bins)
            sums = np.bincount(idx, weights=x, minlength=cls.max_bins)
            keep = counts > 0
            points, counts = sums[keep] / counts[keep], counts[keep]
        n = len(points)
        if n <= k:
            return points
        w = np.concatenate([[0.0], np.cumsum(counts)])
        s1 = np.concatenate([[0.0], np.cumsum(counts * points)])
        s2 = np.concatenate([[0.0], np.cumsum(counts * points**2)])

        def cost(i, j):  # Squared error of points[i:j] as one cluster, i may be an array
            return (s2[j] - s2[i]) - (s1[j] - s1[i]) ** 2 / (w[j] - w[i])

        prev = np.full(n + 1, np.inf)
        prev[0] = 0.0
        splits = []
        for m in range(1, k + 1):
            cur = np.full(n + 1, np.inf)
            arg = np.zeros(n + 1, dtype=np.int64)
            stack = [(m, n, m - 1, n - 1)]  # Fill cur[j] for j in [jlo, jhi], best split in [ilo, ihi]
            while stack:
                jlo, jhi, ilo, ihi = stack.pop()
                if jlo > jhi:
                    continue
                j = (jlo + jhi) // 2
                i = np.arange(ilo, min(ihi, j - 1) + 1)
                c = prev[i] + cost(i, j)
                b = int(np.argmin(c))
                cur[j], arg[j] = c[b], i[b]
                stack.append((jlo, j - 1, ilo, i[b]))
                stack.append((j + 1, jhi, i[b], ihi))
            splits.append(arg)
            prev = cur
        bounds = [n]
        for arg in reversed(splits):
            bounds.append(int(arg[bounds[-1]]))
        bounds = bounds[::-1]
        return np.array([(s1[b] - s1[a]) / (w[b] - w[a]) for a, b in zip(bounds[:-1], bounds[1:])])

    def fitted(self, data):
        """Return a copy whose table is fitted to ``data``."""
        data = np.asarray(data, dtype=np.float64)
        centers = self._kmeans_1d(data, 1 << self.width) if data.size else np.zeros(1)
        max_abs = np.max(np.abs(centers))
        # Integer bits cover the largest entry after rounding, so refitting to quantized data is stable
        decode_integer = int(np.frexp(max_abs)[1]) + 1 if max_abs > 0 else 1
        while True:
            step = 2.0 ** (decode_integer - self.decode_width)
            values = np.unique(np.round(centers / step) * step)
            if np.max(np.abs(values)) < 2.0 ** (decode_integer - 1):
                break
            decode_integer += 1
        if data.size:
            # Keep only entries that some value maps to
            nearest = np.searchsorted((values[1:] + values[:-1]) / 2, data.ravel(), side='left')
            values = values[np.unique(nearest)]
            # Dropping the largest entry can free an integer bit; the entries stay exact with the finer step
            max_abs = np.max(np.abs(values))
            if max_abs > 0:
                decode_integer = min(decode_integer, int(np.frexp(max_abs)[1]) + 1)
        new = copy.copy(self)  # Keep any backend-specific wrapping of the class
        new.decode_integer = decode_integer
        new.values = tuple(float(v) for v in values)
        return new

    def lookup_table(self):
        """Table entry of every index, with indices beyond the fitted entries set to zero."""
        table = np.zeros(1 << self.width)
        table[: len(self.values)] = self.values
        return table

    def encode(self, x):
        """Return the index (int64) of the nearest table entry, ties to the lower entry."""
        if not self.is_fitted:
            raise RuntimeError(f'{self} has not been fitted to any data')
        x = np.asarray(x, dtype=np.float64)
        values = np.array(self.values)
        return np.searchsorted((values[1:] + values[:-1]) / 2, x, side='left').astype(np.int64)

    def decode(self, codes):
        return self.lookup_table()[np.asarray(codes, dtype=np.int64)]

    def quantize(self, x):
        """Round values to the nearest table entry (float64)."""
        return self.decode(self.encode(x))

    def serialize_state(self):
        return {
            'width': self.width,
            'decode_width': self.decode_width,
            'decode_integer': self.decode_integer,
            'values': None if self.values is None else list(self.values),
        }


class UnspecifiedPrecisionType(PrecisionType):
    """
    Class representing an unspecified precision type.

    Instances of this class are expected to be replaced with concrete precision types during conversion.
    """

    def __init__(self):
        super().__init__(width=0, signed=False)


def find_minimum_width(data, signed=True):
    """
    Helper function to find the minimum integer width to express all entries in the data array
    without saturation / overflow.

    Args:
        data (ndarray): Data array.
        signed (bool, optional): Signed or unsigned type. Defaults to ``True.``

    Returns:
        int: Minimum integer width required.
    """
    maxdata = np.amax(np.abs(data))
    if maxdata == 0.0:
        # fringe case (amax(abs(data)) == 0 -> data is uniformly zero)
        return 1

    log2max = np.log2(maxdata)

    iwidth = max(0, int(np.ceil(log2max)))
    if iwidth == int(np.floor(log2max)):  # is a power-of-two integer -> need one extra bit
        iwidth += 1

    if signed:
        # add the sign bit
        iwidth += 1

    return iwidth


# endregion

# region Data type definitions


class NamedType(Serializable):
    """Class representing a named type.

    For convenience, hls4ml gives names to data types used in the generated HLS. This is equivalent to defining types
    in C/C++ like::

        typedef precision name;

    Args:
        name (str): Name given to the type (used in generated C++/HLS).
        precision (PrecisionType): Precision data type.
    """

    def __init__(self, name, precision, **kwargs):
        self.name = name.format(**kwargs)
        self.precision = precision

    def serialize_state(self):
        state = {
            'name': self.name,
            'precision': self.precision.serialize(),
        }
        return state


class CompressedType(NamedType):
    """Class representing a compressed type in COO format.

    Args:
        name (str): Name given to the type (used in generated C++/HLS).
        precision (PrecisionType): Precision data type.
        index_precision (PrecisionType): Precision of the index of COO format.
    """

    def __init__(self, name, precision, index_precision, **kwargs):
        if not name.startswith('compressed_'):
            name = 'compressed_' + name
        super().__init__(name, precision, **kwargs)
        self.index_precision = index_precision

    def serialize_state(self):
        state = super().serialize_state()
        state.update(
            {
                'index_precision': self.index_precision.serialize(),
            }
        )
        return state


class ExponentType(NamedType):
    """Special type used to mark an exponent type, used by the power-of-2 quantizers.

    Args:
        name (str): Name given to the type (used in generated C++/HLS).
        precision (PrecisionType): Precision data type.
    """

    def __init__(self, name, precision, **kwargs):
        if not name.startswith('exponent_'):
            name = 'exponent_' + name
        super().__init__(name, precision, **kwargs)
        self.sign = XnorPrecisionType()


class PackedType(NamedType):
    """A type where multiple elements of the tensor are concatenated and stored as a single element, used by the streaming
    implementations to store elements of the last dimension of a tensor as a single element.

    The tensor of shape ``(H, W, C)`` will be represented as a FIFO stream having ``H * W / n_pack`` elements where each
    element will be a concatenation of ``n_elem * n_pack`` elements of the original tensor.

    Args:
        name (str): Name given to the type (used in generated C++/HLS).
        precision (PrecisionType): Precision data type.
        n_elem (int): Number of packed elements.
        n_pack (int): _description_
    """

    def __init__(self, name, precision, n_elem, n_pack, **kwargs):
        super().__init__(name, precision, **kwargs)
        self.n_elem = n_elem
        if n_pack < 0:
            self.n_pack = -n_pack
            self.unpack = True
        else:
            self.n_pack = n_pack
            self.unpack = False

    def serialize_state(self):
        state = super().serialize_state()
        state.update(
            {
                'n_elem': self.n_elem,
                'n_pack': self.n_pack,
            }
        )
        return state


# endregion

# region Variables


class Variable(Serializable):
    """Base class representing a named multidimensional tensor.

    Args:
        var_name (str): Name of the variable in the generated C++/HLS.
        atype (NamedType): Data type used by the tensor.
    """

    def __init__(self, var_name, atype, **kwargs):
        self.name = var_name.format(**kwargs)
        self.type = atype

    def serialize_state(self):
        state = {
            'name': self.name,
            'type': self.type.serialize(),
        }
        return state


class TensorVariable(Variable):
    """Class representing the output of a layer (like an activation tensor).

    Args:
        shape (list, tuple): Shape of the tensor.
        var_name (str, optional): Name of the variable in the generated C++/HLS. Defaults to ``layer{index}``.
        type_name (str, optional): Name of the data type used (in NamedType). Defaults to ``layer{index}_t``.
        precision (PrecisionType, optional): Precision data type. Defaults to ``None``.
    """

    def __init__(self, shape, var_name='layer{index}', type_name='layer{index}_t', precision=None, **kwargs):
        super().__init__(var_name, NamedType(type_name, precision, **kwargs), **kwargs)
        if isinstance(shape, (list, tuple)):
            self.shape = list(map(int, shape))  # Ensure shape is a list of integers
        else:
            self.shape = [int(shape)]

    def size(self):
        nelem = 1
        for dim in self.shape:
            nelem *= dim
        return nelem

    def size_cpp(self):
        return '*'.join([str(k) for k in self.shape])

    def serialize_state(self):
        state = super().serialize_state()
        state['shape'] = self.shape
        return state

    @classmethod
    def deserialize(cls, state):
        shape = state['shape']
        var_name = state['name']
        type_name = state['type'].name
        precision = state['type'].precision

        return cls(shape, var_name, type_name, precision)


class InplaceTensorVariable(TensorVariable):
    """A ``TensorVariable`` that is just a link to another ``TensorVariable``.

    Args:
        tv (TensorVariable): The tensor variable to link.
        input_var (_type_): The input variable that should be should link to.
    """

    def __init__(self, tv, input_var):
        self.__dict__.update(tv.__dict__)
        self.type = input_var.type
        self.input_var = input_var

    def serialize_state(self):
        state = super().serialize_state()
        state.update(
            {
                'input_var': self.input_var.serialize(),
            }
        )
        return state

    @classmethod
    def deserialize(cls, state):
        tv = TensorVariable.deserialize(state)
        input_var = state['input_var']

        return cls(tv, input_var)


class StructWrapperVariable(TensorVariable):
    def __init__(self, tv):
        self.__dict__.update(tv.__dict__)
        self.struct_name = tv.type.name[:-1] + 'array_t'

    @classmethod
    def deserialize(cls, state):
        tv = TensorVariable.deserialize(state)
        return cls(tv)


class WeightVariable(Variable):
    """Class representing a tensor containing the weights of a layer.

    Precision type of the instance can be modified with the ``update_precision`` method.

    Args:
        var_name (str, optional): Name of the variable in the generated C++/HLS.
        type_name (str, optional): Name of the data type used (in NamedType).
        precision (PrecisionType, optional): Precision data type.
        data (ndarray): The data array.
        quantizer (_type_, optional): Quantizer to apply to the data array. Defaults to ``None``.
    """

    def __init__(self, var_name, type_name, precision, data, quantizer=None, **kwargs):
        super().__init__(var_name, NamedType(type_name, precision, **kwargs), **kwargs)
        self.data = data
        self.nzeros = -1
        self.shape = list(self.data.shape)
        self.data_length = int(np.prod(self.data.shape))
        self.nonzeros = np.count_nonzero(self.data)
        self.nzeros = self.data_length - self.nonzeros
        self.min = np.min(self.data)
        self.max = np.max(self.data)
        self._iterator = None
        self.update_precision(precision)
        self.quantizer = quantizer

    def __iter__(self):
        self._iterator = np.nditer(self.data, order='C')
        return self

    def __next__(self):
        if not self._iterator.finished:
            value = self._iterator[0]
            self._iterator.iternext()
            if isinstance(self.type.precision, CodebookPrecisionType):
                value = int(self.type.precision.encode(value))
            return self.precision_fmt.format(value)
        else:
            raise StopIteration

    next = __next__

    def update_precision(self, new_precision):
        self.type.precision = new_precision
        if isinstance(new_precision, UnspecifiedPrecisionType):
            self.precision_fmt = ''  # Temporarily set precision to undefined value
        elif isinstance(new_precision, (IntegerPrecisionType, XnorPrecisionType, ExponentPrecisionType)):
            self.precision_fmt = '{:.0f}'
        elif isinstance(new_precision, CodebookPrecisionType):
            if not new_precision.is_fitted:
                # Fit the table to these weights; also updates the layer's type, which is self.type
                self.type.precision = new_precision.fitted(self.data)
            self.precision_fmt = '{:d}'  # Table indices
        elif isinstance(new_precision, FixedPrecisionType):
            decimal_spaces = max(0, new_precision.fractional)
            self.precision_fmt = f'{{:.{decimal_spaces}f}}'
        elif isinstance(new_precision, (FloatPrecisionType, StandardFloatPrecisionType)):
            self.precision_fmt = '{:.16f}'  # Not ideal, but should be enough for most cases
        else:
            raise RuntimeError(f'Unexpected new precision type: {new_precision}')

    def serialize_state(self):
        state = super().serialize_state()
        state.update(
            {
                'data': self.data,
                'quantizer': self.quantizer.serialize() if self.quantizer is not None else None,
            }
        )
        return state

    @classmethod
    def deserialize(cls, state):
        var_name = state['name']
        type_name = state['type'].name
        precision = state['type'].precision
        data = state['data']
        quantizer = state['quantizer']

        return cls(var_name, type_name, precision, data, quantizer)


class CompressedWeightVariable(WeightVariable):
    """Class representing a tensor containing the weights of a layer represented in the COO format.

    Args:
        var_name (str, optional): Name of the variable in the generated C++/HLS.
        type_name (str, optional): Name of the data type used (in NamedType).
        precision (PrecisionType, optional): Precision data type.
        data (ndarray): The data array.
        reuse_factor (_type_): The reuse factor used to pad the data array.
        quantizer (_type_, optional): Quantizer to apply to the data array. Defaults to ``None``.
    """

    def __init__(self, var_name, type_name, precision, data, reuse_factor, quantizer=None, **kwargs):
        super().__init__(var_name, type_name, precision, data, quantizer=quantizer, **kwargs)
        self.extra_zeros = 0
        self.data_length = np.prod(data.shape) - self.nzeros
        while self.data_length % reuse_factor != 0:
            self.extra_zeros += 1
            self.data_length += 1
        self.nonzeros = np.prod(data.shape) - self.nzeros + self.extra_zeros

        # Compress the array
        weights = []
        extra_nzero_cnt = self.extra_zeros
        it = np.nditer(data, order='C', flags=['multi_index'])
        max_idx = 0
        while not it.finished:
            val = it[0]
            if not (val == 0 and extra_nzero_cnt < 1):
                if val == 0:
                    extra_nzero_cnt -= 1
                if it.multi_index[0] > max_idx:
                    max_idx = it.multi_index[0]
                if it.multi_index[1] > max_idx:
                    max_idx = it.multi_index[1]
                weights.append([it.multi_index[1], it.multi_index[0], val])
            it.iternext()
        weights.sort()

        index_precision = 32
        if max_idx > 0:
            index_precision = int(np.log2(max_idx) + 1)
        self.type = CompressedType(type_name, precision, IntegerPrecisionType(width=index_precision, signed=False), **kwargs)

        self.data = weights

    def __iter__(self):
        self._iterator = iter(self.data)
        return self

    def __next__(self):
        value = next(self._iterator)
        value_fmt = self.precision_fmt.format(value[2])
        return f'{{{value[1]}, {value[0]}, {value_fmt}}}'

    next = __next__


class ExponentWeightVariable(WeightVariable):
    """WeightVariable for Exponent aka power-of-2 data. The data should already by quantized by the quantizer.

    Args:
        var_name (str, optional): Name of the variable in the generated C++/HLS.
        type_name (str, optional): Name of the data type used (in NamedType).
        precision (PrecisionType, optional): Precision data type.
        data (ndarray): The data array.
        quantizer (_type_, optional): Quantizer to apply to the data array. Defaults to ``None``.
    """

    def __init__(self, var_name, type_name, precision, data, quantizer=None, **kwargs):
        super().__init__(var_name, type_name, precision, data, quantizer, **kwargs)
        self.type = ExponentType(type_name, precision, **kwargs)
        self.shape = list(self.data.shape[:-1])

    def _format(self):
        y = self.data
        # Use an XnorBinary-like representation for the sign
        sign = np.where(y < 0, np.zeros_like(y), np.ones_like(y))
        # Take the logarithm, since this is what we will write to the header
        # for the optimized product using shifts
        y = (np.log2(np.abs(y)) / np.log2(2.0)).astype('int')
        return np.stack((sign, y), axis=-1)

    def __iter__(self):
        data = self._format()
        self._iterator = iter(data.reshape((math.prod(data.shape[:-1]), 2)))
        return self

    def __next__(self):
        value = next(self._iterator)
        value_fmt = self.precision_fmt.format(value[1])
        return f'{{{value[0]}, {value_fmt}}}'

    next = __next__


# endregion

# region Custom source


class Source(Serializable):
    """Class representing generated source code blocks.

    Args:
        code (str): Generated source code.
    """

    def __init__(self, code):
        self.code = code

    def __str__(self):
        return str(self.code)

    def serialize_class_name(self):
        cls = self.__class__
        cls_name = cls.__module__ + '.' + cls.__qualname__
        return cls_name

    def serialize_state(self):
        state = {
            'code': str(self.code),
        }
        return state


# endregion
