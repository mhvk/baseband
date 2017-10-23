"""
Definitions for GSB payloads.

Implements a GSBPayload class used to store payload blocks, and decode to
or encode from a data array.

See http://gmrt.ncra.tifr.res.in/gmrt_hpage/sub_system/gmrt_gsb/index.htm
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from collections import namedtuple

from ..vlbi_base.payload import VLBIPayloadBase


__all__ = ['GSBPayload']


shift40 = np.array([4, 0], np.int8)
shift04 = np.array([0, 4], np.int8)


def decode_4bit(words):
    """Decode 4-bit data.

    For a given int8 byte containing bits 76543210,
    the first sample is in 3210, the second in 7654, and both are interpreted
    as signed 4-bit integers.
    """
    # left_shift(byte[:,np.newaxis], shift40):  [3210xxxx, 76543210]
    split = np.left_shift(words[:, np.newaxis], shift40).ravel()
    # right_shift(..., 4):                      [33333210, 77777654]
    # so least significant bits go first.
    split >>= 4
    return split.astype(np.float32)


def decode_8bit(words):
    """GSB decoder for data stored using 8 bit signed integer.
    """
    return words.astype(np.float32)


def encode_4bit(values):
    b = np.clip(np.round(values), -8, 7).astype(np.int8).reshape(-1, 2)
    b &= 0xf
    b <<= shift04
    return b[:, 0] | b[:, 1]


def encode_8bit(values):
    return np.clip(np.rint(values), -128, 127).astype(np.int8)


class GSBPayload(VLBIPayloadBase):
    """Container for decoding and encoding GSB payloads.

    Parameters
    ----------
    words : ndarray
        Array containg LSB unsigned words (with the right size) that
        encode the payload.
    bps : int
        Number of bits per sample part (i.e., per channel and per real or
        imaginary component).  Default: 2.
    sample_shape : tuple
        Shape of the samples; e.g., (nchan,).  Default: ().
    complex_data : bool
        Whether data is complex or float.  Default: False.
    """

    _encoders = {4: encode_4bit,
                 8: encode_8bit}
    _decoders = {4: decode_4bit,
                 8: decode_8bit}
    _dtype_word = np.int8

    _sample_shape_cls_1thread = namedtuple('SampleShape', 'nchan')
    _sample_shape_cls_nthread = namedtuple('SampleShape', 'nthread, nchan')

    @classmethod
    def _sample_shape_cls(cls, *args):
        if len(args) == 1:
            return cls._sample_shape_cls_1thread(*args)
        else:
            return cls._sample_shape_cls_nthread(*args)

    @classmethod
    def fromfile(cls, fh, payloadsize=None, bps=4, nchan=1,
                 complex_data=False):
        """Read payloads from several threads.

        Parameters
        ----------
        fh : filehandle or tuple of tuple of filehandle
            Handles to the sets of files from which data is read.  The outer
            tuple holds distinct threads, while the inner ones holds parts of
            those threads.  Typically, these are the two polarisations and the
            two parts of each in which phased baseband data are stored.
        payloadsize : int
            Number of bytes to read from each part.
        bps : int
            Number of bits per sample part (i.e., per channel and per real or
            imaginary component).  Default: 4.
        nchan : int
            Number of Fourier channels.  Default: 1.
        complex_data : bool
            Whether data is complex or float.  Default: False.
        """
        if hasattr(fh, 'read'):
            sample_shape = cls._sample_shape_cls(nchan)
            return super(GSBPayload,
                         cls).fromfile(fh, payloadsize=payloadsize,
                                       bps=bps, sample_shape=sample_shape,
                                       complex_data=complex_data)

        nthread = len(fh)
        sample_shape_1thread = cls._sample_shape_cls(nchan)
        payloads = [[super(GSBPayload,
                           cls).fromfile(fh1, payloadsize=payloadsize, bps=bps,
                                         sample_shape=sample_shape_1thread,
                                         complex_data=complex_data)
                     for fh1 in fh_set] for fh_set in fh]
        if nthread == 1:
            words = np.hstack([payload.words for payload in payloads[0]])
        else:
            bpfs = payloads[0][0]._bpfs
            if bpfs % 8:
                raise TypeError('cannot create phased payload: complete sample'
                                ' does not fit in integer number of bytes.')
            words = np.empty((len(payloads[0]),
                              payloads[0][0].words.size * 8 // bpfs,
                              nthread,
                              bpfs // 8), dtype=cls._dtype_word)
            for payload_set, thread in zip(payloads,
                                           words.transpose(2, 0, 1, 3)):
                for payload, part in zip(payload_set, thread):
                    part[:] = payload.words.reshape(-1, bpfs // 8)

        sample_shape_nthread = cls._sample_shape_cls(nthread, nchan)
        return cls(words.ravel(), bps=bps, sample_shape=sample_shape_nthread,
                   complex_data=complex_data)

    def tofile(self, fh):
        try:
            fh.write(self.words.tostring())
        except AttributeError:
            nthread = len(fh)
            assert nthread == self.sample_shape[0]

            words = self.words.reshape(len(fh[0]), -1, nthread,
                                       self._bpfs // nthread // 8)
            for fh_set, thread in zip(fh, words.transpose(2, 0, 1, 3)):
                for fh, part in zip(fh_set, thread):
                    fh.write(part.tostring())
