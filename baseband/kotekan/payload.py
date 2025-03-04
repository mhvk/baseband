# Licensed under the GPLv3 - see LICENSE
"""Payload for DADA format."""
from collections import namedtuple

import numpy as np

from baseband.base.payload import PayloadBase


__all__ = ['KotekanPayload']


levels = np.array(list(range(-8, 8)))
lut = (levels * 1j + levels[:, np.newaxis]).ravel().astype('c8')


def decode_4bit(words):
    return lut[words.view(np.uint8)]


class KotekanPayload(PayloadBase):
    _dtype_word = np.dtype('u1')
    _decoders = {4: decode_4bit}
    _sample_shape_maker = namedtuple('SampleShape', 'npol')
