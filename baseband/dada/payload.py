# Licensed under the GPLv3 - see LICENSE.rst
"""Payload for DADA format."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from collections import namedtuple

from ..vlbi_base.payload import VLBIPayloadBase


__all__ = ['DADAPayload']


def decode_8bit(words):
    return words.view(np.int8, np.ndarray).astype(np.float32)


def encode_8bit(values):
    return np.clip(np.rint(values), -128, 127).astype(np.int8)


class DADAPayload(VLBIPayloadBase):
    """Container for decoding and encoding DADA payloads.

    Parameters
    ----------
    words : ndarray
        Array containg LSB unsigned words (with the right size) that
        encode the payload.
    header : `~baseband.dada.DADAHeader`, optional
        Header that provides information about how the payload is encoded.
        If not give, the following arguments have to be passed in.
    bps : int
        Number of bits per sample part (i.e., per channel and per real or
        imaginary component).  Default: 8.
    sample_shape : tuple
        Shape of the samples; e.g., (nchan,).  Default: ().
    complex_data : bool
        Whether data is complex or float.  Default: False.
    """
    _decoders = {
        8: decode_8bit}
    _encoders = {
        8: encode_8bit}

    _sample_shape_cls = namedtuple('sample_shape', 'npol, nchan')

    def __init__(self, words, header=None, bps=8, sample_shape=(),
                 complex_data=False):
        if header is not None:
            bps = header.bps
            sample_shape = header.sample_shape
            complex_data = header.complex_data
        sample_shape = self._sample_shape_cls(*sample_shape)
        super(DADAPayload, self).__init__(words, sample_shape=sample_shape,
                                          bps=bps, complex_data=complex_data)

    @classmethod
    def fromfile(cls, fh, header=None, memmap=False, payloadsize=None,
                 **kwargs):
        """Read or map encoded data in file.

        Parameters
        ----------
        fh : filehandle
            Handle to the file which will be read or mapped.
        header : `~baseband.dada.DADAHeader`, optional
            If given, used to infer ``payloadsize``, ``bps``, ``sample_shape``,
            and ``complex_data``.  If not given, those have to be passed in.
        memmap : bool, optional
            If `False` (default), read from file.  Otherwise, map the file in
            memory (see `~numpy.memmap`).
        payloadsize : int, optional
            Number of bytes to read (default: as given in ``header``,
            ``cls._size``, or, for mapping, to the end of the file).
        **kwargs
            Additional arguments are passed on to the class initializer. These
            are only needed if ``header`` is not given.
        """
        if payloadsize is None:
            payloadsize = cls._size if header is None else header.payloadsize

        if not memmap:
            return super(DADAPayload, cls).fromfile(fh, header=header,
                                                    payloadsize=payloadsize,
                                                    **kwargs)

        if hasattr(fh, 'memmap'):
            words = fh.memmap(dtype=cls._dtype_word,
                              shape=None if payloadsize is None else
                              (payloadsize // cls._dtype_word.itemsize,))
        else:
            mode = fh.mode.replace('b', '')
            offset = fh.tell()
            words = np.memmap(fh, mode=mode, dtype=cls._dtype_word,
                              offset=offset, shape=None if payloadsize is None
                              else (payloadsize // cls._dtype_word.itemsize,))
            fh.seek(offset + words.size * words.dtype.itemsize)
        return cls(words, header=header, **kwargs)

    @classmethod
    def fromdata(cls, data, bps=2):
        """Encode data as payload.

        Parameters
        ----------
        data : `numpy.ndarray`
            Data to be encoded. The last dimension is taken as the number of
            channels.
        bps : int
            Number of bits per sample to use (for complex data, for real and
            imaginary part separately; default: 2).
        """
        payload = super(DADAPayload, cls).fromdata(data, bps=bps)
        payload.sample_shape = cls._sample_shape_cls(*data.shape[1:])
        return payload
