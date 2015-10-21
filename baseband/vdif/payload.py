"""
Definitions for VLBI VDIF payloads.

Implements a VDIFPayload class used to store payload words, and decode to
or encode from a data array.

For the VDIF specification, see http://www.vlbi.org/vdif
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np

from ..vlbi_base.payload import (VLBIPayloadBase, encode_2bit_real_base,
                                 decoder_levels, DTYPE_WORD)


__all__ = ['init_luts', 'decode_2bit_real', 'encode_2bit_real',
           'decode_4bit_complex', 'encode_4bit_complex', 'VDIFPayload']


def init_luts():
    """Set up the look-up tables for levels as a function of input byte.

    S10. in http://vlbi.org/vdif/docs/VDIF_specification_Release_1.1.1.pdf
    states that samples are encoded by offset-binary, such that all 0 bits is
    lowest and all 1 bits is highest.  I.e., for 2-bit sampling, the order is
    00, 01, 10, 11.
    """
    b = np.arange(256)[:, np.newaxis]
    # 1-bit mode
    i = np.arange(8)
    lut1bit = decoder_levels[1][(b >> i) & 1]
    # 2-bit mode
    i = np.arange(0, 8, 2)
    lut2bit = decoder_levels[2][(b >> i) & 3]
    # 4-bit mode
    i = np.arange(0, 8, 4)
    lut4bit = decoder_levels[4][(b >> i) & 0xf]
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


def encode_2bit_real(values):
    bitvalues = encode_2bit_real_base(values.reshape(-1, 4))
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
    header : `~baseband.vdif.VDIFHeader`, optional
        Information needed to interpret payload.  If not given, the
        following keywords need to be set.

    --- If no `header is given :

    nchan : int, optional
        Number of channels.  Default: 1.
    bps : int, optional
        Bits per complete sample.  Default: 2.
    complex_data : bool
        Complex or float data.  Default: `False`.
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

        Parameters
        ----------
        fh : filehandle
            To read data from.
        header : `~baseband.vdif.VDIFHeader`
            Used to infer the payloadsize, number of channels, bits per sample,
            and whether the data is complex.
        """
        s = fh.read(header.payloadsize)
        if len(s) < header.payloadsize:
            raise EOFError("Could not read full payload.")
        return cls(np.fromstring(s, dtype=DTYPE_WORD), header)

    @classmethod
    def fromdata(cls, data, header):
        """Encode data as payload, using header information.

        Parameters
        ----------
        data : ndarray
            Values to be encoded.
        header : `~baseband.vdif.VDIFHeader`
            Used to infer the encoding, and to verify the number of channels
            and whether the data is complex.
        """
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
