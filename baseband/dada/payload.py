# Licensed under the GPLv3 - see LICENSE.rst
"""Payload for DADA format."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np

from ..vlbi_base.payload import VLBIPayloadBase


__all__ = ['DADAPayload']


def decode_8bit(words):
    return words.view(np.int8, np.ndarray).astype(np.float32)


def encode_8bit(values):
    return np.clip(np.rint(values), -128, 127).astype(np.int8)


class DADAPayload(VLBIPayloadBase):
    _decoders = {
        8: decode_8bit}
    _encoders = {
        8: encode_8bit}

    def __init__(self, words, header=None, bps=2, sample_shape=(),
                 complex_data=False):
        if header is not None:
            bps = header.bps
            sample_shape = header.sample_shape
            complex_data = header.complex_data
        super(DADAPayload, self).__init__(words, sample_shape=sample_shape,
                                          bps=bps, complex_data=complex_data)

    @classmethod
    def fromfile(cls, fh, header=None, memmap=False, payloadsize=None,
                 **kwargs):
        """Memory map encoded data in file.

        Parameters
        ----------
        fh : filehandle
            Handle to the file which will be mapped.
        header : `~baseband.dada.DADAHeader`, optional
            If given, used to infer the payloadsize, bps, sample_shape, and
            whether data is complex.
        memmap : bool, optional
            If `False` (default), read from file.  Otherwise, map the file in
            memory (see `~numpy.memmap`).
        payloadsize : int, optional
            Number of bytes to read (default: as given in ``header``,
            ``cls._size``, or, for memmap, to the end of the file).

        Any other keyword arguments are passed on to the class initialiser.
        """
        if payloadsize is None:
            payloadsize = cls._size if header is None else header.payloadsize

        if not memmap:
            return super(DADAPayload, cls).fromfile(fh, header=header,
                                                    payloadsize=payloadsize,
                                                    **kwargs)

        mode = fh.mode.replace('b', '')

        offset = fh.tell()
        words = np.memmap(fh, mode=mode, dtype=cls._dtype_word, offset=offset,
                          shape=(None if payloadsize is None else
                                 (payloadsize // cls._dtype_word.itemsize,)))
        self = cls(words, header=header, **kwargs)
        fh.seek(offset + self.size)
        return self
