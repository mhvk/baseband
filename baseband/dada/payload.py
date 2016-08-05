# Licensed under the GPLv3 - see LICENSE.rst
"""Payload for DADA format."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np

from ..vlbi_base.payload import VLBIPayloadBase


__all__ = ['DADAPayload']


def decode_8bit_real(words, out=None):
    b = words.view(np.int8).astype(np.float32)
    if out is None:
        return b
    else:
        b.shape = out.shape
        out[:] = b
        return out


def decode_8bit_complex(words, out=None):
    b = words.view(np.int8).astype(np.float32).view(np.complex64)
    if out is None:
        return b
    else:
        b.shape = out.shape
        out[:] = b
        return out


def encode_8bit_real(values):
    return np.clip(np.rint(values), -128, 127).astype(np.int8)


def encode_8bit_complex(values):
    return encode_8bit_real(values.view(values.real.dtype))


class DADAPayload(VLBIPayloadBase):
    _decoders = {
        (8, False): decode_8bit_real,
        (8, True): decode_8bit_complex}
    _encoders = {
        (8, False): encode_8bit_real,
        (8, True): encode_8bit_complex}
