"""
Base definitions for VLBI payloads, used for VDIF and Mark 5B.

Defines a payload class VLBIPayloadBase that can be used to hold the words
corresponding to a frame payload, providing access to the values encoded in
it as a numpy array.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import operator
from functools import reduce
import numpy as np


__all__ = ['VLBIPayloadBase', 'DTYPE_WORD']


DTYPE_WORD = np.dtype('<u4')
"""Dtype for 32-bit unsigned integers, with least signicant byte first."""


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
    bps : int
        Number of bits per sample part (i.e., per channel and per real or
        imaginary component).  Default: 2.
    sample_shape : tuple
        Shape of the samples; e.g., (nchan,).  Default: ().
    complex_data : bool
        Whether data is complex or float.  Default: False.
    """
    # Possible fixed payload size.
    _size = None
    # To be defined by subclasses.
    _encoders = {}
    _decoders = {}

    def __init__(self, words, bps=2, sample_shape=(), complex_data=False):
        self.words = words
        self.sample_shape = sample_shape
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
    def fromdata(cls, data, bps=2):
        """Encode data as a VLBI payload.

        Parameters
        ----------
        data : ndarray
            Data to be encoded. The last dimension is taken as the number of
            channels.
        bps : int
            Number of bits per sample to use (for complex data, for real and
            imaginary part separately; default: 2).
        """
        complex_data = data.dtype.kind == 'c'
        try:
            encoder = cls._encoders[bps, complex_data]
        except KeyError:
            raise ValueError("{0} cannot encode {1} data with {2} bits"
                             .format(cls.__name__, 'complex' if complex_data
                                     else 'real', bps))
        words = encoder(data.ravel()).view(DTYPE_WORD)
        return cls(words, sample_shape=data.shape[1:], bps=bps,
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
                self.bps // (2 if self.complex_data else 1) //
                reduce(operator.mul, self.sample_shape, 1))

    @property
    def shape(self):
        """Shape of the decoded data array (nsample, sample_shape)."""
        return (self.nsample,) + self.sample_shape

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
                self.shape == other.shape and
                self.dtype == other.dtype and
                np.all(self.words == other.words))
