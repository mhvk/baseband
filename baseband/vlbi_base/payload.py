"""
Base definitions for VLBI payloads, used for VDIF and Mark 5B.

Defines a payload class VLBIPayloadBase that can be used to hold the words
corresponding to a frame payload, providing access to the values encoded in
it as a numpy array.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np


__all__ = ['OPTIMAL_2BIT_HIGH', 'TWO_BIT_1_SIGMA', 'FOUR_BIT_1_SIGMA',
           'DTYPE_WORD', 'decoder_levels', 'encode_2bit_real_base',
           'VLBIPayloadBase']

# the high mag value for 2-bit reconstruction
OPTIMAL_2BIT_HIGH = 3.3359
r"""Optimal high value for a 2-bit digitizer for which the low value is 1.

It is chosen such that for a normal distribution in which 68.269% of all values
are at the low level, this is the mean of the others, i.e.,

.. math::

     l = \frac{\int_1^\infty x \exp(-\frac12x^2) dx}
              {\int_0^1 x \exp(-\frac12x^2) dx} \times
         \frac{\int_0^1 \exp(-\frac12x^2)dx}
              {\int_1^\infty \exp(-\frac12x^2) dx}

Note that for this value, the standard deviation is 2.1745.
"""
TWO_BIT_1_SIGMA = 2.1745
"""Optimal level between low and high for the above OPTIMAL_2BIT_HIGH."""
FOUR_BIT_1_SIGMA = 2.95
"""Level for four-bit encoding."""
DTYPE_WORD = np.dtype('<u4')
"""Dtype for 32-bit unsigned integers, with least signicant byte first."""


decoder_levels = {
    1: np.array([-1.0, 1.0], dtype=np.float32),
    2: np.array([-OPTIMAL_2BIT_HIGH, -1.0, 1.0, OPTIMAL_2BIT_HIGH],
                dtype=np.float32),
    4: (np.arange(16) - 8.)/FOUR_BIT_1_SIGMA}
"""Levels for data encoded with different numbers of bits.."""

two_bit_2_sigma = 2 * TWO_BIT_1_SIGMA
clip_low, clip_high = -1.5 * TWO_BIT_1_SIGMA, 1.5 * TWO_BIT_1_SIGMA


def encode_2bit_real_base(values):
    """Encode data using two bits.

    Effectively, get indices such that for ``lv=TWO_BIT_1_SIGMA=2.1745``:
      ================= ======
      Input range       Output
      ================= ======
            value < -lv   0
      -lv < value <  0.   2
       0. < value <  lv   1
       lv < value         3
      ================= ======
    """
    # Optimized for speed by doing calculations in-place, and ensuring that
    # the dtypes match.
    values = np.clip(values, clip_low, clip_high)
    values += two_bit_2_sigma
    bitvalues = np.empty(values.shape, np.uint8)
    return np.floor_divide(values, TWO_BIT_1_SIGMA, out=bitvalues)


class VLBIPayloadBase(object):
    """Container for decoding and encoding VLBI payloads.

    Any subclass should define dictionaries ``_decoders`` and ``_encoders``,
    which hold functions that decode/encode the payload words to/from ndarray.
    These dictionaries are assumed to be indexed by ``(bps, complex_data)``.

    Parameters
    ----------
    words : ndarray
        Array containg LSB unsigned words (with the right size) that
        encode the payload.
    nchan : int
        Number of channels in the data.  Default: 1.
    bps : int
        Number of bits per complete sample.  Default: 2.
    complex_data : bool
        Whether data is complex or float.  Default: False.
    """
    # Possible fixed payload size.
    _size = None
    # To be defined by subclasses.
    _encoders = {}
    _decoders = {}

    def __init__(self, words, nchan=1, bps=2, complex_data=False):
        self.words = words
        self.nchan = nchan
        self.bps = bps
        self.complex_data = complex_data
        if self._size is not None and self._size != self.size:
            raise ValueError("Encoded data should have length {0}"
                             .format(self._size))

    @classmethod
    def fromfile(cls, fh, *args, **kwargs):
        """Read payload from file handle and decode it into data.

        Parameters
        ----------
        fh : filehandle
            Handle to the file from which data is read
        payloadsize : int
            Number of bytes to read (default: as given in ``cls._size``.

        Any other (keyword) arguments are passed on to the class initialiser.
        """
        payloadsize = kwargs.pop('payloadsize', cls._size)
        if payloadsize is None:
            raise ValueError("Payloadsize should be given as an argument "
                             "if no default is defined on the class.")
        s = fh.read(payloadsize)
        if len(s) < payloadsize:
            raise EOFError("Could not read full payload.")
        return cls(np.fromstring(s, dtype=DTYPE_WORD), *args, **kwargs)

    def tofile(self, fh):
        """Write VLBI payload to filehandle."""
        return fh.write(self.words.tostring())

    @classmethod
    def fromdata(cls, data, bps=None):
        """Encode data as a VLBI payload.

        Parameters
        ----------
        data : ndarray
            Data to be encoded. The last dimension is taken as the number of
            channels.
        bps : int
            Number of bits per sample to use (for complex data, for real and
            imaginary part together; default: 2 for real, 4 for complex).
        """
        complex_data = data.dtype.kind == 'c'
        if bps is None:
            bps = 4 if complex_data else 2
        encoder = cls._encoders[bps, complex_data]
        words = encoder(data.ravel())
        return cls(words, nchan=data.shape[-1], bps=bps,
                   complex_data=complex_data)

    def todata(self, data=None):
        """Decode the payload.

        Parameters
        ----------
        data : ndarray or None
            If given, used to decode the payload into.  It should have the
            right size to store it.  Its shape is not changed.
        """
        decoder = self._decoders[self.bps, self.complex_data]
        out = decoder(self.words, out=data)
        return out.reshape(self.shape) if data is None else data

    data = property(todata, doc="Decoded payload.")

    def __array__(self):
        """Interface to arrays."""
        return self.data

    @property
    def nsample(self):
        """Number of samples in the payload."""
        return (len(self.words) * (self.words.dtype.itemsize * 8) //
                self.bps // self.nchan)

    @property
    def shape(self):
        """Shape of the decoded data array (nsample, nchan)."""
        return (self.nsample, self.nchan)

    @property
    def dtype(self):
        """Type of the decoded data array."""
        return np.dtype(np.complex64 if self.complex_data else np.float32)

    @property
    def size(self):
        """Size in bytes of payload."""
        return len(self.words) * self.words.dtype.itemsize

    def __eq__(self, other):
        return (type(self) is type(other) and
                np.all(self.words == other.words))
