# Licensed under the GPLv3 - see LICENSE
"""Payload for DADA format."""
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
