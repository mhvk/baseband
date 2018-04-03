# Licensed under the GPLv3 - see LICENSE
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
    words : `~numpy.ndarray`
        Array containg LSB unsigned words (with the right size) that
        encode the payload.
    header : `~baseband.dada.DADAHeader`
        Header that provides information about how the payload is encoded.
        If not given, the following arguments have to be passed in.
    bps : int, optional
        Number of bits per sample part (i.e., per channel and per real or
        imaginary component).  Default: 8.
    sample_shape : tuple, optional
        Shape of the samples; e.g., (nchan,).  Default: ().
    complex_data : bool, optional
        Whether data are complex.  Default: `False`.
    """
    _decoders = {
        8: decode_8bit}
    _encoders = {
        8: encode_8bit}

    _sample_shape_maker = namedtuple('SampleShape', 'npol, nchan')

    def __init__(self, words, header=None, sample_shape=(), bps=8,
                 complex_data=False):
        if header is not None:
            sample_shape = header.sample_shape
            bps = header.bps
            complex_data = header.complex_data
        super(DADAPayload, self).__init__(words, sample_shape=sample_shape,
                                          bps=bps, complex_data=complex_data)

    @classmethod
    def fromfile(cls, fh, header=None, memmap=False, payload_nbytes=None,
                 **kwargs):
        """Read or map encoded data in file.

        Parameters
        ----------
        fh : filehandle
            Handle to the file which will be read or mapped.
        header : `~baseband.dada.DADAHeader`, optional
            If given, used to infer ``payload_nbytes``, ``bps``,
            ``sample_shape``, and ``complex_data``.  If not given, those have
            to be passed in.
        memmap : bool, optional
            If `False` (default), read from file.  Otherwise, map the file in
            memory (see `~numpy.memmap`).
        payload_nbytes : int, optional
            Number of bytes to read (default: as given in ``header``,
            ``cls._nbytes``, or, for mapping, to the end of the file).
        **kwargs
            Additional arguments are passed on to the class initializer. These
            are only needed if ``header`` is not given.
        """
        if payload_nbytes is None:
            payload_nbytes = (cls._nbytes if header is None
                              else header.payload_nbytes)

        if not memmap:
            return super(DADAPayload, cls).fromfile(
                fh, header=header, payload_nbytes=payload_nbytes, **kwargs)

        if hasattr(fh, 'memmap'):
            words = fh.memmap(dtype=cls._dtype_word,
                              shape=None if payload_nbytes is None else
                              (payload_nbytes // cls._dtype_word.itemsize,))
        else:
            mode = fh.mode.replace('b', '')
            offset = fh.tell()
            words = np.memmap(
                fh, mode=mode, dtype=cls._dtype_word, offset=offset,
                shape=(None if payload_nbytes is None
                       else (payload_nbytes // cls._dtype_word.itemsize,)))
            fh.seek(offset + words.size * words.dtype.itemsize)
        return cls(words, header=header, **kwargs)
