import numpy as np
from ..vlbi_helpers import VLBIPayloadBase

# the high mag value for 2-bit reconstruction
OPTIMAL_2BIT_HIGH = 3.3359
FOUR_BIT_1_SIGMA = 2.95
DTYPE_WORD = np.dtype('<u4')


# Some duplication with mark4.py here: lut2bit = mark4.lut2bit1
# Though lut1bit = -mark4.lut1bit, so perhaps not worth combining.
def init_luts():
    """Set up the look-up tables for levels as a function of input byte."""
    lut2level = np.array([-1.0, 1.0], dtype=np.float32)
    lut4level = np.array([-OPTIMAL_2BIT_HIGH, 1.0, -1.0, OPTIMAL_2BIT_HIGH],
                         dtype=np.float32)
    b = np.arange(256)[:, np.newaxis]
    # 1-bit mode
    l = np.arange(8)
    lut1bit = lut2level[(b >> l) & 1]
    # 2-bit mode
    s = np.arange(0, 8, 2)
    lut2bit = lut4level[(b >> s) & 3]
    return lut1bit, lut2bit

lut1bit, lut2bit = init_luts()


# def decode_1bit(frame, nvlbichan):
#     return lut1bit[frame].reshape(-1, nvlbichan)


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


shift2bit = np.arange(0, 8, 2).astype(np.uint8)
reorder = np.array([0, 2, 1, 3], dtype=np.uint8)


def encode_2bit_real(values):
    # Effectively, get indices such that:
    #       value < -2. : 0
    # -2. < value <  0. : 2
    #  0. < value <  2. : 1
    #  2. < value       : 3
    # Optimized for speed by doing most calculations in-place, and ensuring
    # that the dtypes match.
    values = np.clip(values.reshape(-1, 4), -3., 3.)
    values += 4.
    bitvalues = np.empty(values.shape, np.uint8)
    bitvalues = np.floor_divide(values, 2., out=bitvalues)
    # swap 1 & 2
    reorder.take(bitvalues, out=bitvalues)
    bitvalues <<= shift2bit
    return np.bitwise_or.reduce(bitvalues, axis=-1).view(DTYPE_WORD)


class Mark5BPayload(VLBIPayloadBase):

    _size = 2500 * 4
    _encoders = {(2, False): encode_2bit_real}
    _decoders = {(2, False): decode_2bit_real}

    def __init__(self, words, nchan=1, bps=2):
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
        """
        super(Mark5BPayload, self).__init__(words, nchan, bps,
                                            complex_data=False)

    @classmethod
    def fromdata(cls, data, bps=2):
        """Encode data as payload, using a given bits per second.

        It is assumed that the last dimension is the number of channels.
        """
        if data.dtype.kind == 'c':
            raise ValueError("Mark5B format does not support complex data.")
        encoder = cls._encoders[bps, False]
        return cls(encoder(data.ravel()), nchan=data.shape[-1], bps=bps)
