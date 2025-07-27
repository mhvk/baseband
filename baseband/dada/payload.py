# Licensed under the GPLv3 - see LICENSE
"""Payload for DADA format."""
from collections import namedtuple

import numpy as np

from ..base.payload import PayloadBase


__all__ = ['DADAPayload', "MKBFPayload"]


def decode_8bit(words):
    return words.view(np.int8, np.ndarray).astype(np.float32)


def encode_8bit(values):
    return np.clip(np.rint(values), -128, 127).astype(np.int8)


class DADAPayload(PayloadBase):
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
    _memmap = True
    _sample_shape_maker = namedtuple('SampleShape', 'npol, nchan')

    def __new__(cls, words, *, header=None, **kwargs):
        # Override instrument if header is given.
        if header is not None and header.get("INSTRUMENT") == "MKBF":
            cls = MKBFPayload
        return super().__new__(cls)


class MKBFPayload(DADAPayload):
    """Container for decoding and encoding MKBF DADA payloads.

    Subclass of `~baseband.dada.DADAPayload` that takes into account
    that the samples are organized in heaps of 256 samples, with order
    (nheap, npol, nchan, nsub=256).

    Some information on the instrument writing it can be found in
    `Van der Byl et al. 2021 <https://doi.org/10.1117/1.JATIS.8.1.011006>`_

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Change view of words to indicate one element containts 256
        # samples for each part of the sample_shape (this also ensures
        # bpw=bits-per-word is correct in _item_to_slices).
        self._dtype_word = np.dtype(
            [("heaps",
              [("real", f"int{self.bps}"), ("imag", f"int{self.bps}")],
              self.sample_shape + (256,))])
        self.words = self.words.view(self._dtype_word)

    def _decode(self, words, data_slice=()):
        words = np.moveaxis(words["heaps"], -1, 1).ravel().reshape(
            -1, *self.sample_shape)[data_slice]
        return super()._decode(words)

    def _encode(self, data):
        data = np.moveaxis(data.reshape(-1, 256, *self.sample_shape), 1, -1)
        return super()._encode(data)

    def __getitem__(self, item=()):
        words_slice, data_slice = self._item_to_slices(item)
        return self._decode(self.words[words_slice], data_slice)

    data = property(__getitem__, doc="Full decoded payload.")
