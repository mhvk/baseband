# Licensed under the GPLv3 - see LICENSE
"""
Definitions for VLBI VDIF payloads.

Implements a VDIFPayload class used to store payload words, and decode to
or encode from a data array.

See the `VDIF specification page <http://www.vlbi.org/vdif>`_ for payload
specifications.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from collections import namedtuple

from ..vlbi_base.payload import VLBIPayloadBase
from ..vlbi_base.encoding import (encode_2bit_base, encode_4bit_base,
                                  decoder_levels, decode_8bit, encode_8bit)

__all__ = ['init_luts', 'decode_2bit', 'decode_4bit', 'encode_2bit',
           'encode_4bit', 'VDIFPayload']


def init_luts():
    """Sets up the look-up tables for levels as a function of input byte.

    Returns
    -------
    lut1bit : `~numpy.ndarray`
        Look-up table for decoding bytes to 1-bit samples.
    lut2bit : `~numpy.ndarray`
        As `lut1bit1`, but for 2-bit samples.
    lut4bit : `~numpy.ndarray`
        As `lut1bit1`, but for 4-bit samples.

    Notes
    -----

    Look-up tables are two-dimensional arrays whose first axis is indexed
    by byte value (in uint8 form) and whose second axis represents sample
    temporal order.  Table values are decoded sample values.  Sec. 10 in
    the `VDIF Specification
    <http://vlbi.org/vdif/docs/VDIF_specification_Release_1.1.1.pdf>`_
    states that samples are encoded by offset-binary, such that all 0
    bits is lowest and all 1 bits is highest.  I.e., for 2-bit sampling,
    the order is 00, 01, 10, 11.  These are decoded using
    `~baseband.vlbi_base.encoding.decoder_levels`.

    For example, the 2-bit sample sequence ``-1, -1, 1, 1`` is encoded
    as ``0b10100101`` (or ``165`` in uint8 form).  To translate this back
    to sample values, access ``lut2bit`` using the byte as the key::

        >>> lut2bit[0b10100101]
        array([-1., -1.,  1.,  1.], dtype=float32)
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


def decode_2bit(words):
    """Decodes data stored using 2 bits per sample."""
    b = words.view(np.uint8)
    return lut2bit.take(b, axis=0)


shift2bit = np.arange(0, 8, 2).astype(np.uint8)


def encode_2bit(values):
    """Encodes values using 2 bits per sample, packing the result into bytes.
    """
    bitvalues = encode_2bit_base(values.reshape(-1, 4))
    bitvalues <<= shift2bit
    return np.bitwise_or.reduce(bitvalues, axis=-1)


def decode_4bit(words):
    """Decodes data stored using 4 bits per sample."""
    b = words.view(np.uint8)
    return lut4bit.take(b, axis=0)


shift04 = np.array([0, 4], np.uint8)


def encode_4bit(values):
    """Encodes values using 4 bits per sample, packing the result into bytes.
    """
    b = encode_4bit_base(values).reshape(-1, 2)
    b <<= shift04
    return b[:, 0] | b[:, 1]


class VDIFPayload(VLBIPayloadBase):
    """Container for decoding and encoding VDIF payloads.

    Parameters
    ----------
    words : `~numpy.ndarray`
        Array containg LSB unsigned words (with the right size) that
        encode the payload.
    header : `~baseband.vdif.VDIFHeader`
        If given, used to infer the number of channels, bps, and whether
        the data are complex.
    nchan : int, optional
        Number of channels, used if ``header`` is not given.  Default: 1.
    bps : int, optional
        Bits per elementary sample, used if ``header`` is not given.
        Default: 2.
    complex_data : bool, optional
        Whether the data are complex, used if ``header`` is not given.
        Default: `False`.
    """
    _decoders = {2: decode_2bit,
                 4: decode_4bit,
                 8: decode_8bit}

    _encoders = {2: encode_2bit,
                 4: encode_4bit,
                 8: encode_8bit}

    _sample_shape_maker = namedtuple('SampleShape', 'nchan')

    def __init__(self, words, header=None, nchan=1, bps=2, complex_data=False):
        if header is not None:
            nchan = header.nchan
            bps = header.bps
            complex_data = header['complex_data']
            self._nbytes = header.payload_nbytes
            if header.edv == 0xab:  # Mark5B payload
                from ..mark5b import Mark5BPayload
                self._decoders = Mark5BPayload._decoders
                self._encoders = Mark5BPayload._encoders
                if complex_data:
                    raise ValueError("VDIF/Mark5B payload cannot be complex.")
        super(VDIFPayload, self).__init__(words, sample_shape=(nchan,),
                                          bps=bps, complex_data=complex_data)

    @classmethod
    def fromfile(cls, fh, header):
        """Read payload from filehandle and decode it into data.

        Parameters
        ----------
        fh : filehandle
            To read data from.
        header : `~baseband.vdif.VDIFHeader`
            Used to infer the payload size, number of channels, bits per
            sample, and whether the data are complex.
        """
        s = fh.read(header.payload_nbytes)
        if len(s) < header.payload_nbytes:
            raise EOFError("could not read full payload.")
        return cls(np.frombuffer(s, dtype=cls._dtype_word), header)

    @classmethod
    def fromdata(cls, data, header=None, bps=2, edv=None):
        """Encode data as payload, using header information.

        Parameters
        ----------
        data : `~numpy.ndarray`
            Values to be encoded.
        header : `~baseband.vdif.VDIFHeader`, optional
            If given, used to infer the encoding, and to verify the number of
            channels and whether the data are complex.
        bps : int, optional
            Bits per elementary sample, used if ``header`` is not given.
            Default: 2.
        edv : int, optional
            Should be given if ``header`` is not given and the payload is
            encoded as Mark 5 data (i.e., ``edv=0xab``).
        """
        nchan = data.shape[-1]
        complex_data = (data.dtype.kind == 'c')
        if header is not None:
            if header.nchan != nchan:
                raise ValueError("header is for {0} channels but data has {1}"
                                 .format(header.nchan, data.shape[-1]))
            if header['complex_data'] != complex_data:
                raise ValueError("header is for {0} data but data are {1}"
                                 .format(*(('complex' if c else 'real') for c
                                           in (header['complex_data'],
                                               complex_data))))
            bps = header.bps
            edv = header.edv

        if edv == 0xab:  # Mark5B payload
            from ..mark5b import Mark5BPayload
            encoder = Mark5BPayload._encoders[bps]
        else:
            encoder = cls._encoders[bps]

        if complex_data:
            data = data.view((data.real.dtype, (2,)))
        words = encoder(data).ravel().view(cls._dtype_word)
        return cls(words, header, nchan=nchan, bps=bps,
                   complex_data=complex_data)
