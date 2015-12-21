# Licensed under the GPLv3 - see LICENSE.rst
import io
import numpy as np
from astropy.tests.helper import pytest
import astropy.units as u
from astropy.time import Time
from .. import gsb
from ..gsb.payload import decode_4bit_real, encode_4bit_real


class TestGSB(object):
    def test_header(self):
        line = ('2014 01 20 02 28 10 0.811174 '
                '2014 01 20 02 28 10 0.622453760 5049 1')
        header = gsb.GSBHeader(tuple(line.split()))
        assert header.mode == 'phased'
        assert header['pc'] == line[:28]
        assert header['gps'] == line[29:60]
        assert header['seq_nr'] == 5049
        assert header['sub_int'] == 1
        assert abs(header.pc_time -
                   Time('2014-01-19T20:58:10.811174')) < 1.*u.ns
        assert header.gps_time == header.time
        assert abs(header.time -
                   Time('2014-01-19T20:58:10.622453760')) < 1.*u.ns
        assert header.mutable is False
        with pytest.raises(TypeError):
            header['sub_int'] = 0

        with io.StringIO() as s:
            header.tofile(s)
            s.seek(0)
            assert s.readline().strip() == line
            s.seek(0)
            header2 = gsb.GSBHeader.fromfile(s)
        assert header == header2
        assert header2.mutable is False
        header3 = gsb.GSBHeader.fromkeys(**header)
        assert header3 == header
        assert header3.mutable is True
        with pytest.raises(KeyError):
            gsb.GSBHeader.fromkeys(extra=1, **header)
        with pytest.raises(KeyError):
            kwargs = dict(header)
            kwargs.pop('seq_nr')
            gsb.GSBHeader.fromkeys(**kwargs)
        # Try initialising with properties instead of keywords.
        header4 = gsb.GSBHeader.fromvalues(time=header.time,
                                           pc_time=header.pc_time,
                                           seq_nr=header['seq_nr'],
                                           sub_int=header['sub_int'])
        assert header4 == header
        assert header4.mutable is True
        header5 = header.copy()
        assert header5 == header
        assert header5.mutable is True
        header5['seq_nr'] = header['seq_nr'] + 1
        assert header5['seq_nr'] == header['seq_nr'] + 1
        assert header5 != header
        header5.time = Time('2014-01-20T05:30:00')
        assert header5['gps'] == '2014 01 20 11 00 00 0.000000000'
        header5['gps'] = '2014 01 20 11 00 00.000000000 0'
        with pytest.raises(ValueError):
            header5.time

        # Quick checks on rawdump mode
        line = '2015 04 27 18 45 00 0.000000240'
        header6 = gsb.GSBHeader(line.split())
        assert header6.mode == 'rawdump'
        assert header6['pc'] == line
        assert abs(header6.time -
                   Time('2015-04-27T13:15:00.000000240')) < 1. * u.ns
        header7 = gsb.GSBHeader.fromkeys(**header6)
        assert header7 == header6
        header8 = gsb.GSBHeader.fromvalues(mode='rawdump', **header6)
        assert header8 == header6
        with pytest.raises(TypeError):
            gsb.GSBHeader.fromvalues(**header6)
        with pytest.raises(TypeError):
            gsb.GSBHeader(None)

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
