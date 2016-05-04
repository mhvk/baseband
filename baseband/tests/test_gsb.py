# Licensed under the GPLv3 - see LICENSE.rst
import io
import numpy as np
from astropy.tests.helper import pytest
import astropy.units as u
from astropy.time import Time
from astropy.tests.helper import assert_quantity_allclose
from .. import gsb
from ..gsb.payload import decode_4bit_real, encode_4bit_real


class TestGSB(object):
    def setup(self):
        self.rawdump_ts = '2015 04 27 18 45 00 0.000000240'
        self.phased_ts = ('2014 01 20 02 28 10 0.811174 '
                          '2014 01 20 02 28 10 0.622453760 5049 1')
        self.data = np.clip(np.round(np.random.uniform(-8.5, 7.5, size=2048)),
                            -8, 7).reshape(-1, 1)

    def test_header(self):
        header = gsb.GSBHeader(tuple(self.phased_ts.split()))
        assert header.mode == 'phased'
        assert header['pc'] == self.phased_ts[:28]
        assert header['gps'] == self.phased_ts[29:60]
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
            assert s.readline().strip() == self.phased_ts
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
        header6 = gsb.GSBHeader(self.rawdump_ts.split())
        assert header6.mode == 'rawdump'
        assert header6['pc'] == self.rawdump_ts
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
        # Check that recovering with the actual header type works as well.
        header9 = type(header).fromkeys(**header)
        assert header9 == header
        with pytest.raises(KeyError):
            type(header).fromkeys(**header6)
        header10 = type(header6).fromkeys(**header6)
        assert header10 == header6
        with pytest.raises(KeyError):
            type(header6).fromkeys(**header)

    def test_header_seek_offset(self):
        header = gsb.GSBHeader(tuple(self.phased_ts.split()))
        header_size = len(self.phased_ts) + 1
        assert header.seek_offset(1) == header_size
        # seq_nr=5049, sub_int=1
        n_1000_0 = 1000 * 8 - (5049 * 8 + 1)
        offset_to_1000_0 = header.seek_offset(n_1000_0)
        assert offset_to_1000_0 == n_1000_0 * header_size
        # Go to 999 7
        assert header.seek_offset(n_1000_0 - 1) == (
            (n_1000_0 - 1) * header_size + 1)
        # Go to 100 0
        assert header.seek_offset(n_1000_0 - 900 * 8) == (
            (n_1000_0 - 900 * 8) * header_size + 900 * 8)
        # Go to 99 7
        assert header.seek_offset(n_1000_0 - 900 * 8 - 1) == (
            (n_1000_0 - 900 * 8 - 1) * header_size + 900 * 8 + 2)

    def test_header_non_gmrt(self):
        header = gsb.GSBHeader(tuple(self.phased_ts.split()),
                               utc_offset=0.*u.hr)
        assert abs(header.pc_time -
                   Time('2014-01-20T02:28:10.811174')) < 1.*u.ns
        assert header.gps_time == header.time
        assert abs(header.time -
                   Time('2014-01-20T02:28:10.622453760')) < 1.*u.ns

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
        payload1 = gsb.GSBPayload.fromdata(self.data, bps=4)
        assert np.all(payload1.data == self.data)
        payload2 = gsb.GSBPayload.fromdata(self.data[:1024], bps=8)
        assert np.all(payload2.data == self.data[:1024])
        cmplx = self.data[::2] + 1j * self.data[1::2]
        payload3 = gsb.GSBPayload.fromdata(cmplx, bps=4)
        assert np.all(payload3.data == cmplx)
        assert np.all(payload3.words == payload1.words)
        payload4 = gsb.GSBPayload.fromdata(cmplx[:512], bps=8)
        assert np.all(payload4.data == cmplx[:512])
        assert np.all(payload4.words == payload2.words)
        channelized = self.data.reshape(-1, 512)
        payload5 = gsb.GSBPayload.fromdata(channelized, bps=4)
        assert payload5.shape == channelized.shape
        assert np.all(payload5.words == payload1.words)
        with io.BytesIO() as s:
            payload1.tofile(s)
            s.seek(0)
            payload6 = gsb.GSBPayload.fromfile(s, bps=4,
                                               payloadsize=payload1.size)
        assert payload6 == payload1

    def test_rawdump_frame(self):
        header1 = gsb.GSBHeader(self.rawdump_ts.split())
        frame1 = gsb.GSBFrame.fromdata(self.data, header1, bps=4)
        assert np.all(frame1.data == self.data)
        assert frame1.size == len(self.data) // 2
        frame2 = gsb.GSBFrame(frame1.header, frame1.payload)
        assert frame2 == frame1
        with io.StringIO() as sh, io.BytesIO() as sp:
            frame1.tofile(sh, sp)
            sh.seek(0)
            sp.seek(0)
            frame3 = gsb.GSBFrame.fromfile(sh, sp, bps=4,
                                           payloadsize=frame1.size)
        assert frame3 == frame1

    def test_phased_frame(self):
        data = self.data.reshape((1,) + self.data.shape)
        header1 = gsb.GSBHeader(self.phased_ts.split())
        frame1 = gsb.GSBFrame.fromdata(data, header1, bps=4)
        assert frame1.shape == data.shape
        assert np.all(frame1.data == data)
        assert frame1.size == data.size // 2
        cmplx = data[:, ::2] + 1j * data[:, 1::2]
        frame2 = gsb.GSBFrame.fromdata(cmplx, header1, bps=4)
        assert frame2.dtype.kind == 'c'
        assert np.all(frame2.data == cmplx)
        assert frame2.valid is True
        frame2.valid = False
        assert frame2.valid is False
        assert np.all(frame2.data == 0.)
        frame2.valid = True
        assert frame2.valid is True
        assert np.all(frame2.data == cmplx)
        with io.StringIO() as sh, io.BytesIO() as sp0, io.BytesIO() as sp1:
            frame1.tofile(sh, ((sp0, sp1),))
            assert sp0.tell() == frame1.size // 2
            assert sp1.tell() == frame1.size // 2
            for s in sh, sp0, sp1:
                s.flush()
                s.seek(0)

            frame3 = gsb.GSBFrame.fromfile(sh, ((sp0, sp1),), bps=4,
                                           payloadsize=data.size // 4)
        assert frame3 == frame1
        # Two polarisations, 16 channels
        twopol = cmplx.reshape(2, -1, 16)
        frame4 = gsb.GSBFrame.fromdata(twopol, header1, bps=4)
        assert frame4.size == frame1.size  # number of bytes doesn't change.
        assert frame4.shape == twopol.shape
        assert np.all(frame4.data == twopol)
        with io.StringIO() as sh, \
                io.BytesIO() as sp0, io.BytesIO() as sp1, \
                io.BytesIO() as sp2, io.BytesIO() as sp3:
            frame4.tofile(sh, ((sp0, sp1), (sp2, sp3)))
            for s in sh, sp0, sp1, sp2, sp3:
                s.flush()
                s.seek(0)

            frame5 = gsb.GSBFrame.fromfile(sh, ((sp0, sp1), (sp2, sp3)),
                                           bps=4, payloadsize=data.size // 8)
        assert frame5 == frame4

    def test_timestamp_io(self):
        header = gsb.GSBHeader(tuple(self.phased_ts.split()))
        with io.BytesIO() as s, gsb.open(s, 'wt') as fh_w:
            fh_w.write_timestamp(header)
            fh_w.flush()
            s.seek(0)
            with gsb.open(s, 'rt') as fh_r:
                header2 = fh_r.read_timestamp()
            assert header == header2

    def test_rawfile_io(self):
        payload = gsb.GSBPayload.fromdata(self.data, bps=4)
        with io.BytesIO() as s, gsb.open(s, 'wb') as fh_w:
            fh_w.write_payload(payload)
            fh_w.flush()
            s.seek(0)
            with gsb.open(s, 'rb') as fh_r:
                payload2 = fh_r.read_payload(payload.size, payload.nchan,
                                             payload.bps, payload.complex_data)
            assert payload == payload2

    def test_raw_stream(self):
        header = gsb.GSBHeader(self.rawdump_ts.split())
        payload = gsb.GSBPayload.fromdata(self.data, bps=4)
        with io.BytesIO() as s, gsb.open(s, 'wt') as sh, io.BytesIO() as sp:
            sh.write_timestamp(header)
            sh.flush()
            payload.tofile(sp)
            sp.flush()
            s.seek(0)
            sp.seek(0)
            with gsb.open(s, mode='rs', raw=sp,
                          samples_per_frame=payload.nsample,
                          frames_per_second=1) as fh_r:
                assert fh_r.header0 == header
                data = fh_r.read(len(self.data))
                assert fh_r.tell() == len(data)
                assert_quantity_allclose(fh_r.tell(unit=u.s), 1. * u.s)
            assert np.all(data == self.data.ravel())

        with io.BytesIO() as sh, io.BytesIO() as sp, gsb.open(
                sh, 'ws', raw=sp, sample_rate=4096*u.Hz,
                samples_per_frame=payload.nsample,
                header=header) as fh_w:
            fh_w.write(self.data.ravel())
            fh_w.write(self.data.ravel())
            assert fh_w.tell() == 2 * len(self.data)
            assert_quantity_allclose(fh_w.tell(unit=u.s), 1. * u.s)
            fh_w.flush()
            sh.seek(0)
            sp.seek(0)
            with gsb.open(sh, mode='rs', raw=sp,
                          samples_per_frame=payload.nsample) as fh_r:
                assert fh_r.header0 == header
                assert np.isclose(fh_r.frames_per_second, 2)
                data = fh_r.read(len(self.data) * 2)
            assert np.all(data.reshape(2, -1) == self.data.ravel())

    @pytest.mark.parametrize('bps', (4, 8))
    def test_phased_stream(self, bps):
        header = gsb.GSBHeader(self.phased_ts.split())
        # Two polarisations, 16 channels
        cmplx = self.data[::2] + 1j * self.data[1::2]
        twopol = cmplx.reshape(-1, 2, 16)
        with io.BytesIO() as sh, \
                io.BytesIO() as sp0, io.BytesIO() as sp1, \
                io.BytesIO() as sp2, io.BytesIO() as sp3, \
                gsb.open(sh, 'ws', raw=((sp0, sp1), (sp2, sp3)),
                         bps=bps, sample_rate=128*u.Hz,
                         samples_per_frame=twopol.shape[0] // 2,
                         nchan=twopol.shape[2], nthread=twopol.shape[1],
                         complex_data=True, header=header) as fh_w:
            fh_w.write(twopol)
            fh_w.write(twopol[::-1])
            assert fh_w.tell() == twopol.shape[0] * 2
            assert_quantity_allclose(fh_w.tell(unit=u.s), 0.5 * u.s)
            assert fh_w._header['seq_nr'] == fh_w.header0['seq_nr'] + 3
            fh_w.flush()
            assert sp0.tell() == 1024 * bps // 8
            for fh in sh, sp0, sp1, sp2, sp3:
                fh.seek(0)
            with gsb.open(sh, mode='rs', raw=((sp0, sp1), (sp2, sp3)),
                          bps=bps, samples_per_frame=twopol.shape[0] // 2,
                          nchan=twopol.shape[2]) as fh_r:
                assert fh_r.header0 == header
                assert np.isclose(fh_r.frames_per_second, 8.)
                data = fh_r.read(twopol.shape[0] * 2)
                assert fh_r.tell() == twopol.shape[0] * 2
                assert_quantity_allclose(fh_r.tell(unit=u.s), 0.5 * u.s)
                assert sp0.tell() == 1024 * bps // 8
            assert np.all(data[:twopol.shape[0]] == twopol)
            assert np.all(data[twopol.shape[0]:] == twopol[::-1])
