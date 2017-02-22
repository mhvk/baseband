# Licensed under the GPLv3 - see LICENSE.rst
"""Payload for GUPPI format."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np

from ..vlbi_base.payload import VLBIPayloadBase


__all__ = ['GUPPIPayload']


def decode_8bit(words):
    return words.view(np.int8, np.ndarray).astype(np.float32)


def encode_8bit(values):
    return np.clip(np.rint(values), -128, 127).astype(np.int8)


class GUPPIPayload(VLBIPayloadBase):
    """Container for decoding and encoding GUPPI payloads.

    Parameters
    ----------
    words : ndarray
        Array containg LSB unsigned words (with the right size) that
        encode the payload.
    header : `~baseband.dada.GUPPIHeader`, optional
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

    def __init__(self, words, header=None, bps=8, sample_shape=(),
                 complex_data=False, time_ordered=True):
        if header is not None:
            bps = header.bps
            sample_shape = header.sample_shape
            complex_data = header.complex_data
            time_ordered = header.time_ordered
        super(GUPPIPayload, self).__init__(words, sample_shape=sample_shape,
                                           bps=bps, complex_data=complex_data)
        self.time_ordered = time_ordered

    @classmethod
    def fromfile(cls, fh, header=None, memmap=False, payloadsize=None,
                 **kwargs):
        """Read or map encoded data in file.

        Parameters
        ----------
        fh : filehandle
            Handle to the file which will be read or mapped.
        header : `~baseband.dada.GUPPIHeader`, optional
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
            return super(GUPPIPayload, cls).fromfile(fh, header=header,
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

    def __getitem__(self, item=()):
        if not self.time_ordered:
            return super(GUPPIPayload, self).__getitem__(item)

        # time-ordered means the channels are stored sequentially as blocks.
        # for simplicity, just decode the lot.
        data = super(GUPPIPayload, self).__getitem__()
        # reshape to what it should have been in the first place.
        # TODO: be less blunt!!
        data = data.reshape(data.shape[1], data.shape[0],
                            data.shape[2]).transpose(1, 0, 2)
        return data.__getitem__(item)

    data = property(__getitem__, doc="Full decoded payload.")

