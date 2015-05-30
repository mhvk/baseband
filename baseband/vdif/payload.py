import numpy as np


# the high mag value for 2-bit reconstruction
OPTIMAL_2BIT_HIGH = 3.3359
FOUR_BIT_1_SIGMA = 2.95
DTYPE_WORD = np.dtype('<u4')


class VDIFPayload(object):
    def __init__(self, words, header=None,
                 nchan=1, bps=2, complex_data=False):
        """Container for decoding and encoding VDIF payloads.

        Parameters
        ----------
        words : ndarray
            Array containg LSB unsigned words (with the right size) that
            encode the payload.
        header : VDIFHeader or None
            Information needed to interpret payload.

        If ``header`` is not given, one needs to pass the following:

        nchan : int
            Number of channels in the data.  Default: 1.
        bps : int
            Number of bits per complete sample.  Default: 2.
        complex_data : bool
            Whether data is complex or float.  Default: False.
        """
        if header is not None:
            nchan = header.nchan
            bps = header.bps
            complex_data = header['complex_data']

        self.words = words
        self.nchan = nchan
        self.bps = bps
        self.complex_data = complex_data
        self.nsample = len(words) * (32 // self.bps) // self.nchan

    @classmethod
    def frombytes(cls, raw, *args, **kwargs):
        """Set paiload by interpreting bytes."""
        return cls(np.fromstring(raw, dtype=DTYPE_WORD), *args, **kwargs)

    def tobytes(self):
        """Convert payload to bytes."""
        return self.words.tostring()

    @classmethod
    def fromfile(cls, fh, header):
        """Read payload from file handle and decode it into data."""
        self = cls.frombytes(fh.read(header.payloadsize), header)
        if len(self.data) < header.payloadsize:
            raise EOFError("Could not read full payload.")
        return self

    def tofile(self, fh):
        return fh.write(self.tobytes())

    @classmethod
    def fromdata(cls, data, header=None, bps=2):
        complex_data = data.dtype.kind == 'c'
        if header is not None:
            bps = header.bps

        encoder = ENCODERS[bps, complex_data]
        return cls(encoder(data.ravel()),
                   nchan=data.shape[-1], bps=bps, complex_data=complex_data)

    def todata(self, data=None):
        """Decode the payload.

        Parameters
        ----------
        data : ndarray or None
            If given, used to decode the payload into.  It should have the
            right size to store it.  Its shape is not changed.
        """
        decoder = DECODERS[self.bps, self.complex_data]
        out = decoder(self.words, out=data)
        return out.reshape(self.shape) if data is None else data

    data = property(todata, doc="Decode the payload.")

    @property
    def shape(self):
        return (self.nsample, self.nchan)

    @property
    def dtype(self):
        return np.complex64 if self.complex_data else np.float32


def init_luts():
    """Set up the look-up tables for levels as a function of input byte."""
    lut2level = np.array([-1.0, 1.0], dtype=np.float32)
    lut4level = np.array([-OPTIMAL_2BIT_HIGH, -1.0, 1.0, OPTIMAL_2BIT_HIGH],
                         dtype=np.float32)
    lut16level = (np.arange(16) - 8.)/FOUR_BIT_1_SIGMA

    b = np.arange(256)[:, np.newaxis]
    # 1-bit mode
    i = np.arange(8)
    lut1bit = lut2level[(b >> i) & 1]
    # 2-bit mode
    i = np.arange(0, 8, 2)
    lut2bit = lut4level[(b >> i) & 3]
    # 4-bit mode
    i = np.arange(0, 8, 4)
    lut4bit = lut16level[(b >> i) & 0xf]
    return lut1bit, lut2bit, lut4bit

lut1bit, lut2bit, lut4bit = init_luts()


# Decoders keyed by bits_per_sample, complex_data:
def decode_2bit_real(words, out=None):
    b = words.view(np.uint8)
    if out is None:
        return lut2bit.take(b, axis=0).ravel()
    else:
        outf4 = out.reshape(-1, 4)
        assert outf4.base is out or outf4.base is out.base
        lut2bit.take(b, axis=0, out=outf4)
        return out


def decode_4bit_complex(words, out=None):
    if out is None:
        return lut2bit.take(words.view(np.uint8),
                            axis=0).ravel().view(np.complex64)
    else:
        outf4 = out.reshape(-1, 2).view(np.float32)
        assert outf4.base is out
        lut2bit.take(words.view(np.uint8), axis=0, out=outf4)
        return out


DECODERS = {
    (2, False): decode_2bit_real,
    (4, True): decode_4bit_complex
}


shift2bit = np.arange(0, 8, 2).astype(np.uint8)


def encode_2bit_real(values):
    # Effectively, get indices such that:
    #       value < -2. : 0
    # -2. < value <  0. : 1
    #  0. < value <  2. : 2
    #  2. < value       : 3
    # Optimized for speed by doing most calculations in-place, and ensuring
    # that the dtypes match.
    values = np.clip(values.reshape(-1, 4), -3., 3.)
    values += 4.
    bitvalues = np.empty(values.shape, np.uint8)
    bitvalues = np.floor_divide(values, 2., out=bitvalues)
    bitvalues <<= shift2bit
    return np.bitwise_or.reduce(bitvalues, axis=-1).view(DTYPE_WORD)


def encode_4bit_complex(values):
    return encode_2bit_real(values.view(values.real.dtype)).view(DTYPE_WORD)


ENCODERS = {
    (2, False): encode_2bit_real,
    (4, True): encode_4bit_complex
}
