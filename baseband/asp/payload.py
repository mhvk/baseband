import numpy as np
from ..base.payload import PayloadBase
from ..base.utils import fixedvalue
from collections import namedtuple


__all__ = ['ASPPayload']


def decode_8bit(words):
    return words.view(np.int8, np.ndarray).astype(np.float32)


def encode_8bit(values):
    return np.clip(np.rint(values), -128, 127).astype(np.int8)


NPOL = 2
NDIM = 2


class ASPPayload(PayloadBase):
    _encoders = {8: encode_8bit}
    _decoders = {8: decode_8bit}
    _dtype_word = np.dtype('<u4')  # 2 pol complex

    _sample_shape_maker = namedtuple('SampleShape', 'npol')

    # Define init just to change defaults.
    def __init__(self, words, *, header=None,
                 sample_shape=(2,), bps=8, complex_data=True):
        super().__init__(words, header=header, sample_shape=sample_shape,
                         bps=bps, complex_data=complex_data)

    @fixedvalue
    def complex_data(cls):
        """ASP data is always complex."""
        return True
