import numpy as np
from ..base.payload import PayloadBase
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
    _dtype_word = np.dtype('int8')

    _sample_shape_maker = namedtuple('SampleShape', 'npol')

    # sample_shape here corresponds to polarization
    def __init__(self, words, header=None, sample_shape=(2,), **kwargs):
        assert(header is not None)
        # ASP will always hold complex data and bps is fixed at 8bit
        super(ASPPayload, self).__init__(
            words, sample_shape=sample_shape,
            bps=8, complex_data=True, **kwargs)

    @classmethod
    def fromfile(cls, fh, header=None, payload_nbytes=None, **kwargs):
        # the payload size is variable, in principle, so we always need
        # a header to inform us.
        # the header field is actually a block header
        assert(header is not None)

        npts = header['NPtsSend'][0]
        payload_count = NPOL * NDIM * npts
        nbytes_read = payload_count * cls._dtype_word.itemsize

        buf = fh.read(nbytes_read)
        if(len(buf) < nbytes_read):
            raise EOFError('reached EOF while reading ASPPayload')
        words = np.frombuffer(buf, dtype=cls._dtype_word, count=payload_count)
        return cls(words, header=header, **kwargs)
