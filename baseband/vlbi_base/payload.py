# Helper functions for VLBI readers (VDIF, Mark5B).
import numpy as np

# the high mag value for 2-bit reconstruction
OPTIMAL_2BIT_HIGH = 3.3359
FOUR_BIT_1_SIGMA = 2.95
DTYPE_WORD = np.dtype('<u4')


class VLBIPayloadBase(object):

    _size = None
    _encoders = {}
    _decoders = {}

    def __init__(self, words, nchan=1, bps=2, complex_data=False):
        """Container for decoding and encoding VDIF payloads.

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
        self.words = words
        self.nchan = nchan
        self.bps = bps
        self.complex_data = complex_data
        self.nsample = len(words) * (32 // self.bps) // self.nchan
        if self._size is not None and self._size != self.size:
            raise ValueError("Encoded data should have length {0}"
                             .format(self._size))

    @classmethod
    def frombytes(cls, raw, *args, **kwargs):
        """Set paiload by interpreting bytes."""
        return cls(np.fromstring(raw, dtype=DTYPE_WORD), *args, **kwargs)

    def tobytes(self):
        """Convert payload to bytes."""
        return self.words.tostring()

    @classmethod
    def fromfile(cls, fh, *args, **kwargs):
        """Read payload from file handle and decode it into data.

        Parameters
        ----------
        fh : filehandle
            Handle to the file from which data is read
        payloadsize : int
            Number of bytes to read (default: as given in ``cls._payloadsize``.

        Any other (keyword) arguments are passed on to the class initialiser.
        """
        payloadsize = kwargs.pop('payloadsize', cls._size)
        if payloadsize is None:
            raise ValueError("Payloadsize should be given as an argument "
                             "if no default is defined on the class.")
        s = fh.read(payloadsize)
        if len(s) < payloadsize:
            raise EOFError("Could not read full payload.")
        return cls.frombytes(s, *args, **kwargs)

    def tofile(self, fh):
        return fh.write(self.tobytes())

    @classmethod
    def fromdata(cls, data, bps=2):
        """Encode data as payload, using a given bits per second.

        It is assumed that the last dimension is the number of channels.
        """
        complex_data = data.dtype.kind == 'c'
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

    data = property(todata, doc="Decode the payload.")

    @property
    def shape(self):
        return (self.nsample, self.nchan)

    @property
    def dtype(self):
        return np.dtype(np.complex64 if self.complex_data else np.float32)

    @property
    def size(self):
        """Size in bytes of payload."""
        return len(self.words) * DTYPE_WORD.itemsize

    def __eq__(self, other):
        return (type(self) is type(other) and
                np.all(self.words == other.words))
