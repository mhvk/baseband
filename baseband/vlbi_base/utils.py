# Licensed under the GPLv3 - see LICENSE
from operator import index
from math import gcd

import numpy as np


__all__ = ['lcm', 'bcd_decode', 'bcd_encode', 'byte_array', 'CRC', 'CRCStack']


def lcm(a, b):
    """Calculate the least common multiple of a and b."""
    return abs(a * b) // gcd(a, b)


def bcd_decode(value):
    try:
        # Far faster than my routine for scalars
        return int('{:x}'.format(index(value)))
    except TypeError as exc:  # Might still be an array
        try:
            assert value.dtype.kind in 'iu'
        except Exception:
            raise exc

    d = np.arange(value.itemsize * 2)
    digits = (value[:, np.newaxis] >> d*4) & 0xf
    if digits.max() > 9:
        bad = value[(digits > 9).any(1)][0]
        raise ValueError("invalid BCD encoded value {0}={1}."
                         .format(bad, hex(bad)))
    return (digits * 10**d).sum(1)


def bcd_encode(value):
    try:
        # Far faster than my routine for scalars
        return int('{:d}'.format(index(value)), base=16)
    except TypeError as exc:  # Might still be an array
        try:
            assert value.dtype.kind in 'iu'
        except Exception:
            raise exc

    d = np.arange(value.itemsize * 2)
    digits = (value[:, np.newaxis] // 10**d) % 10
    return (digits << d*4).sum(1)


def byte_array(pattern):
    """Convert the pattern to a byte array.

    Parameters
    ----------
    pattern : ~numpy.ndarray, bytes, int, or iterable of int
        Pattern to convert.  If a `~numpy.ndarray` or `bytes` instance,
        a byte array view is taken.  If an (iterable of) int, the integers
        need to be unsigned 32 bit and will be interpreted as little-endian.

    Returns
    -------
    byte_array : `~numpy.ndarray` of byte
        With any elements of pattern stored in little-endian order.
    """
    if isinstance(pattern, (np.ndarray, bytes)):
        # Quick turn-around for input that is OK already:
        return np.atleast_1d(pattern).view('u1')

    pattern = np.array(pattern, ndmin=1)
    if (pattern.dtype.kind not in 'uif'
            or pattern.min() < 0
            or pattern.max() >= 1 << 32):
        raise ValueError('values have to fit in 32 bit unsigned int.')
    return pattern.astype('<u4').view('u1')


class CRC:
    """Cyclic Redundancy Check.

    See https://en.wikipedia.org/wiki/Cyclic_redundancy_check

    Once initialised, the instance can be used as a function that calculates
    the CRC, or one can use the ``check`` method to verify that the CRC in
    the lower bits of a value is correct.

    Parameters
    ----------
    polynomial : int
        Binary encoded CRC divisor. For instance, that used by Mark 5B headers
        is 0x18005, or x^16 + x^15 + x^2 + 1.

    See Also
    --------
    baseband.vlbi_base.utils.CRCStack :
        for calculating CRC on arrays where each entry represents a bit.
    """

    def __init__(self, polynomial):
        self.polynomial = index(polynomial)

    def __len__(self):
        return self.polynomial.bit_length() - 1

    def __call__(self, stream):
        """Calculate CRC for the given stream.

        Parameters
        ----------
        stream : int or array of unsigned int
            The integer (or array of integers) to calculate the CRC for.

        Returns
        -------
        crc : int or array
            If an array, the crc will have the same dtype as the input stream.
        """
        return self._crc(stream, extend=True)

    def check(self, stream):
        """Check that the CRC at the end of athe stream is correct.

        Parameters
        ----------
        stream : int or array of unsigned int
            For an integer, the value is the stream to check the CRC for.
            For arrays, the dimension is treated as the index into the bits.
            A single stream would thus be of type `bool`. Unsigned integers
            represent multiple streams. E.g., for a 64-track Mark 4 header,
            the stream would be an array of ``np.uint64`` words.

        Returns
        -------
        ok : bool
             `True` if the calculated CRC is all zero (which should be the
             case if the CRC at the end of the stream is correct).
        """
        return self._crc(stream) == 0

    def _crc(self, stream, extend=False):
        try:
            scalar = index(stream)
        except TypeError:
            return self._crc_array(stream, extend=extend)
        else:
            return self._crc_scalar(scalar, extend=extend)

    def _crc_scalar(self, scalar, extend=False):
        """Internal function to calculate the CRC for a scalar."""
        # This routine uses bit_length() to find where the highest
        # remaining set bit is, thus skipping all the zeros where nothing
        # needs to be done.  It is about 15 times faster than a 1-element
        # array and can handle arbitrarily long integers.
        nbp = self.polynomial.bit_length()
        nbs = scalar.bit_length()
        while nbs >= nbp:
            scalar ^= self.polynomial << nbs-nbp
            nbs = scalar.bit_length()

        if extend:
            # Do the further iterations to *calculate* a CRC.
            return self._crc_scalar(scalar << nbp-1)

        return scalar

    def _crc_array(self, array, extend=False):
        """Internal function to calculate the CRC for an array."""
        # Here, the array contains individual entries for which the
        # CRC should be calculated.
        array = np.array(array, copy=True, dtype='u8')
        nbp = self.polynomial.bit_length()
        nbs = index(array.max()).bit_length()
        while nbs >= nbp:
            mask = (array & (1 << nbs-1)).astype(bool).astype(array.dtype)
            mask *= self.polynomial << nbs-nbp
            array ^= mask
            nbs = int(array.max()).bit_length()

        if extend:
            return self._crc_array(array << nbp-1)

        return array


class CRCStack(CRC):
    """Cyclic Redundancy Check for a bitstream.

    See https://en.wikipedia.org/wiki/Cyclic_redundancy_check

    Once initialised, the instance can be used as a function that calculates
    the CRC, or one can use the ``check`` method to verify that the CRC at
    the end of a stream is correct.

    This class is specifically for arrays in which multiple bit streams
    occupy different bit levels, and the dimension is treated as the
    index into the bits.  A single stream would thus be of type `bool`.
    Unsigned integers represent multiple streams. E.g., for a 64-track
    Mark 4 header, the stream would be an array of ``np.uint64`` words.

    Parameters
    ----------
    polynomial : int
        Binary encoded CRC divisor. For instance, that used by Mark 4 headers
        is 0x180f, or x^12 + x^11 + x^3 + x^2 + x + 1.

    See Also
    --------
    baseband.vlbi_base.utils.CRC :
        for calculating CRC for a single value or an array of values.
    """
    def __init__(self, polynomial):
        super().__init__(polynomial)
        binary_str = '{:b}'.format(self.polynomial)
        self._npol_array = np.array([-int(bit) for bit in binary_str],
                                    dtype='i1')

    def check(self, stream):
        return np.all(self._crc(stream) == 0)

    def _crc(self, stream, extend=False):
        """Internal function to calculate the CRC for a bit stream."""
        ncrc = len(self)
        if extend:
            stream = np.hstack((stream, np.zeros(ncrc, stream.dtype)))
        else:
            stream = stream.copy()

        # Make an all-bits-one for each set item.
        pol_array = self._npol_array.astype(stream.dtype)
        for i, bits in enumerate(stream[:-ncrc]):
            stream[i:i+ncrc+1] ^= (bits & pol_array)

        return stream[-ncrc:]
