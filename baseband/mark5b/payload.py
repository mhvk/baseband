# Licensed under the GPLv3 - see LICENSE
"""
Definitions for VLBI Mark 5B payloads.

Implements a Mark5BPayload class used to store payload words, and decode to
or encode from a data array.

For the specification, see
https://www.haystack.mit.edu/tech/vlbi/mark5/docs/Mark%205B%20users%20manual.pdf
"""
from collections import namedtuple

import numpy as np

from ..vlbi_base.payload import VLBIPayloadBase
from ..vlbi_base.encoding import encode_2bit_base, decoder_levels


__all__ = ['init_luts', 'decode_1bit', 'decode_2bit',
           'encode_1bit', 'encode_2bit',
           'Mark5BPayload']


# Some duplication with mark4.py here: lut2bit = mark4.lut2bit1
# Though lut1bit = -mark4.lut1bit, so perhaps not worth combining.
def init_luts():
    """Set up the look-up tables for levels as a function of input byte.

    For 1-bit mode, one has just the sign bit:
     === =====
      s  value
     === =====
      0  +1
      1  -1
     === =====

    For 2-bit mode, there is a sign and a magnitude, which encode:
     === === ===== =====
      m   s  value s*2+m
     === === ===== =====
      0   0  -Hi    0
      0   1  +1     2
      1   0  -1     1
      1   1  +Hi    3
     === === ===== =====

    Note that the sign bit is flipped for 1-bit mode from what one might
    expect from the 2-bit encodings (at least, as implemented in mark5access;
    the docs are rather unclear).  For more details, see Table 13 in
    https://library.nrao.edu/public/memos/vlba/up/VLBASU_13.pdf
    and
    https://www.haystack.mit.edu/tech/vlbi/mark5/docs/Mark%205B%20users%20manual.pdf
    Appendix A: sign always on even bit stream (0, 2, 4, ...), and magnitude
    on adjacent odd stream (1, 3, 5, ...).

    In the above table, the last column is the index in the linearly increasing
    table of levels (``decoder_levels[2]``).
    """
    b = np.arange(256)[:, np.newaxis]
    sl = np.arange(8)
    # For 1-bit modes, like for Mark 4, if set, sign=-1, so need to get item 0.
    # With this, one reproduces mark5access m5d, but must say that from the
    # documentation, it is not obvious this is correct.
    lut1bit = decoder_levels[1][((b >> sl) & 1) ^ 1]
    # 2-bit mode: sign bit in lower position thatn magnitude bit
    # ms=00,01,10,11 = -Hi, 1, -1, Hi (lut
    s = np.arange(0, 8, 2)  # 0, 2, 4, 6
    m = s + 1               # 1, 3, 5, 7
    sl = (((b >> s) & 1) << 1) + ((b >> m) & 1)
    lut2bit = decoder_levels[2][sl]
    return lut1bit, lut2bit


lut1bit, lut2bit = init_luts()


def decode_1bit(words):
    b = words.view(np.uint8)
    return lut1bit.take(b, axis=0)


shift1bit = np.arange(0, 8).astype(np.uint8)


def encode_1bit(values):
    """Encodes values using 1 bit per sample, packing the result into bytes."""
    bitvalues = np.signbit(values.reshape(-1, 8)).view(np.uint8)
    return np.packbits(bitvalues[:, ::-1])


def decode_2bit(words):
    b = words.view(np.uint8)
    return lut2bit.take(b, axis=0)


shift2bit = np.arange(0, 8, 2).astype(np.uint8)
reorder = np.array([0, 2, 1, 3], dtype=np.uint8)


def encode_2bit(values):
    bitvalues = encode_2bit_base(values.reshape(-1, 4))
    # swap 1 & 2
    reorder.take(bitvalues, out=bitvalues)
    bitvalues <<= shift2bit
    return np.bitwise_or.reduce(bitvalues, axis=-1)


encode_2bit.__doc__ = encode_2bit_base.__doc__


class Mark5BPayload(VLBIPayloadBase):
    """Container for decoding and encoding VDIF payloads.

    Parameters
    ----------
    words : `~numpy.ndarray`
        Array containg LSB unsigned words (with the right size) that
        encode the payload.
    nchan : int, optional
        Number of channels.   Default: 1.
    bps : int, optional
        Bits per elementary sample.  Default: 2.
    """

    _nbytes = 2500 * 4
    _encoders = {1: encode_1bit,
                 2: encode_2bit}
    _decoders = {1: decode_1bit,
                 2: decode_2bit}

    _sample_shape_maker = namedtuple('SampleShape', 'nchan')

    def __init__(self, words, nchan=1, bps=2, complex_data=False):
        if complex_data:
            raise ValueError("Mark5B format does not support complex data.")

        super().__init__(words, sample_shape=(nchan,),
                         bps=bps, complex_data=False)

    @classmethod
    def fromdata(cls, data, bps=2):
        """Encode data as payload, using a given number of bits per sample.

        It is assumed that the last dimension is the number of channels.
        """
        if data.dtype.kind == 'c':
            raise ValueError("Mark5B format does not support complex data.")
        encoder = cls._encoders[bps]
        words = encoder(data).view(cls._dtype_word)
        return cls(words, nchan=data.shape[-1], bps=bps)
