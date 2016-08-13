# Licensed under the GPLv3 - see LICENSE.rst
"""Payload for DADA format."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np

from ..vlbi_base.payload import VLBIPayloadBase


__all__ = ['DADAPayload']


def decode_8bit(words):
    return words.view(np.int8).astype(np.float32)


def encode_8bit(values):
    return np.clip(np.rint(values), -128, 127).astype(np.int8)


class DADAPayload(VLBIPayloadBase):
    _decoders = {
        8: decode_8bit}
    _encoders = {
        8: encode_8bit}
