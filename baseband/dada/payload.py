# Licensed under the GPLv3 - see LICENSE
"""Payload for DADA format."""
import math
from collections import namedtuple

import numpy as np

from ..base.payload import PayloadBase


__all__ = ['DADAPayload']


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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Change view of words to indicate one element containts 256
        # samples for each part of the sample_shape (this also ensures
        # bpw=bits-per-word is correct in _item_to_slices).
        self.words = self.words.view(
            [("samples", "u2", self.sample_shape + (256,))]
        )

    def _samples_first(self, words):
        """Reorder words to have samples first."""
        return np.reshape(
            np.moveaxis(words["samples"], -1, 1),
            (-1,) + self.sample_shape, copy=True,
        )

    def __getitem__(self, item=()):
        if item == () or item == slice(None):
            words= self._samples_first(self.words)

        else:
            words_slice, data_slice = self._item_to_slices(item)
            words = self._samples_first(self.words[words_slice])[data_slice]

        data = self._decoders[self._coder](words)

        return data.view(self.dtype) if self.complex_data else data

    data = property(__getitem__, doc="Full decoded payload.")
