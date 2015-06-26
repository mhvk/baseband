"""
Definitions for VLBI VDIF payloads.

Implements a VDIFPayload class used to store payload words, and decode to
or encode from a data array.

For the VDIF specification, see http://www.vlbi.org/vdif
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np

from ..vlbi_base import (VLBIPayloadBase, OPTIMAL_2BIT_HIGH, TWO_BIT_1_SIGMA,
                         FOUR_BIT_1_SIGMA, DTYPE_WORD)


def init_luts():
    """Set up the look-up tables for levels as a function of input byte.

    S10. in http://vlbi.org/vdif/docs/VDIF_specification_Release_1.1.1.pdf
    states that samples are encoded by offset-binary, such that all 0 bits is
    lowest and all 1 bits is highest.  I.e., for 2-bit sampling, the order is
    00, 01, 10, 11.
    """
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
    b = words.view(np.uint8)
    if out is None:
        return lut2bit.take(b, axis=0).ravel().view(np.complex64)
    else:
        outf4 = out.reshape(-1, 2).view(np.float32)
        assert outf4.base is out or outf4.base is out.base
        lut2bit.take(words.view(np.uint8), axis=0, out=outf4)
        return out


shift2bit = np.arange(0, 8, 2).astype(np.uint8)
two_bit_2_sigma = 2 * TWO_BIT_1_SIGMA
clip_low, clip_high = -1.5 * TWO_BIT_1_SIGMA, 1.5 * TWO_BIT_1_SIGMA


def encode_2bit_real(values):
    """Encode data using two bits.

    Effectively, get indices such that for lv=TWO_BIT_1_SIGMA=2.1745:
            value < -lv : 0
      -lv < value <  0. : 1
       0. < value <  lv : 2
       2. < value       : 3
    """
    # Optimized for speed by doing most calculations in-place, and ensuring
    # that the dtypes match.
    values = np.clip(values.reshape(-1, 4), clip_low, clip_high)
    values += two_bit_2_sigma
    bitvalues = np.empty(values.shape, np.uint8)
    bitvalues = np.floor_divide(values, TWO_BIT_1_SIGMA, out=bitvalues)
    bitvalues <<= shift2bit
    return np.bitwise_or.reduce(bitvalues, axis=-1).view(DTYPE_WORD)


def encode_4bit_complex(values):
    return encode_2bit_real(values.view(values.real.dtype))


class VDIFPayload(VLBIPayloadBase):
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
    _decoders = {(2, False): decode_2bit_real,
                 (4, True): decode_4bit_complex}

    _encoders = {(2, False): encode_2bit_real,
                 (4, True): encode_4bit_complex}

    def __init__(self, words, header=None,
                 nchan=1, bps=2, complex_data=False):
        if header is not None:
            nchan = header.nchan
            bps = header.bps
            complex_data = header['complex_data']
            self._size = header.payloadsize
            if header.edv == 0xab:  # Mark5B payload
                from ..mark5b import Mark5BPayload
                self._decoders = Mark5BPayload._decoders
                self._encoders = Mark5BPayload._encoders
                if complex_data:
                    raise ValueError("VDIF/Mark5B payload cannot be complex.")
        super(VDIFPayload, self).__init__(words, nchan, bps, complex_data)

    @classmethod
    def fromfile(cls, fh, header):
        """Read payload from file handle and decode it into data.

        The payloadsize, number of channels, bits per sample, and whether
        data are complex are all taken from the header.
        """
        s = fh.read(header.payloadsize)
        if len(s) < header.payloadsize:
            raise EOFError("Could not read full payload.")
        return cls(np.fromstring(s, dtype=DTYPE_WORD), header)

    @classmethod
    def fromdata(cls, data, header):
        """Encode data as payload, using header information."""
        if header.nchan != data.shape[-1]:
            raise ValueError("Header is for {0} channels but data has {1}"
                             .format(header.nchan, data.shape[-1]))
        if header['complex_data'] != (data.dtype.kind == 'c'):
            raise ValueError("Header is for {0} data but data is {1}"
                             .format(('complex' if c else 'real') for c
                                     in (header['complex_data'],
                                         data.dtype.kind == 'c')))
        if header.edv == 0xab:  # Mark5B payload
            from ..mark5b import Mark5BPayload
            encoder = Mark5BPayload._encoders[header.bps,
                                              header['complex_data']]
        else:
            encoder = cls._encoders[header.bps, header['complex_data']]
        words = encoder(data.ravel())
        return cls(words, header)
