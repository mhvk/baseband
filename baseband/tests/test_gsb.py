# Licensed under the GPLv3 - see LICENSE.rst
import io
import numpy as np
from astropy.tests.helper import pytest
from .. import gsb
from ..gsb.payload import decode_4bit_real, encode_4bit_real


class TestGSB(object):
    def test_decoding(self):
        """Check that 4-bit encoding works."""
        areal = np.arange(-8, 8)
        b = encode_4bit_real(areal)
        assert np.all(b.view(np.uint8) ==
                      np.array([0x98, 0xba, 0xdc, 0xfe,
                                0x10, 0x32, 0x54, 0x76]))
        d = decode_4bit_real(b)
        assert np.all(d == areal)

    def test_payload(self):
        data = np.clip(np.round(np.random.uniform(-8.5, 7.5, size=2048)),
                       -8, 7)
        payload1 = gsb.GSBPayload.fromdata(data, bps=4)
        assert np.all(payload1.data == data)
        payload2 = gsb.GSBPayload.fromdata(data[:1024], bps=8)
        assert np.all(payload2.data == data[:1024])
        cmplx = data[::2] + 1j * data[1::2]
        payload3 = gsb.GSBPayload.fromdata(cmplx, bps=4)
        assert np.all(payload3.data == cmplx)
        assert np.all(payload3.words == payload1.words)
        payload4 = gsb.GSBPayload.fromdata(cmplx[:512], bps=8)
        assert np.all(payload4.data == cmplx[:512])
        assert np.all(payload4.words == payload2.words)
        channelized = data.reshape(-1, 512)
        payload5 = gsb.GSBPayload.fromdata(channelized, bps=4)
        assert payload5.shape == channelized.shape
        assert np.all(payload5.words == payload1.words)
        with io.BytesIO() as s:
            payload1.tofile(s)
            s.seek(0)
            payload6 = gsb.GSBPayload.fromfile(s, bps=4,
                                               payloadsize=payload1.size)
        assert payload6 == payload1
