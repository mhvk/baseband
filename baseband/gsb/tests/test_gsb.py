# Licensed under the GPLv3 - see LICENSE.rst
import io
import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.tests.helper import assert_quantity_allclose
from ... import gsb
from ..payload import decode_4bit, encode_4bit
from ...data import SAMPLE_GSB_HEADER as SAMPLE_HEADER


# Test on 2016-AUG-19, using GMRT crab data
# sshfs login.scinet.utoronto.ca
# /scratch2/p/pen/franzk/data/GMRT/ddtb144_04oct2015/a gmrt
# ipython  # not python3 for scintellometry
# from astropy import units as u
# from baseband import gsb; from scintellometry.io import gmrt
# gmrt_base = 'gmrt/crab-04-10-15.raw'
# tsf = gmrt_base + '.timestamp'
# raws = [[gmrt_base + '.Pol-' + pol + part + '.dat' for part in ('1', '2')]
#         for pol in ('L', 'R')]
# fh_gsb = gsb.open(tsf, raw=raws, mode='rs', bps=8, samples_per_frame=2**14,
#                   complex_data=True, nchan=512)
# fh_gmrt = gmrt.GMRTPhasedData(tsf, raws[0], 2**23, 512, 200./3*u.MHz,
#                               170*u.MHz, True)
# Note: GMRTPhasedData can only read single polarisation at a time.
# fh_gsb.seek(2**15)
# np.all(fh_gmrt.seek_record_read(2**25, 2048) == fh_gsb.read(2)[:, 0])


class TestGSB(object):
    def setup(self):
        self.rawdump_ts = '2015 04 27 18 45 00 0.000000240'
        self.data = np.clip(np.round(np.random.uniform(-8.5, 7.5, size=2048)),
                            -8, 7).reshape(-1, 1)

    def test_header(self):
        with gsb.open(SAMPLE_HEADER, 'rt') as fh:
            header = gsb.GSBHeader.fromfile(fh, verify=True)
            fh.seek(0)
            h_raw = fh.readline().strip()
        assert header.mode == 'phased'
        assert header['pc'] == h_raw[:28]
        assert header['gps'] == h_raw[29:60]
        assert header['seq_nr'] == 9980
        assert header['sub_int'] == 4
        assert abs(header.pc_time -
                   Time('2013-07-25T19:32:51.733965')) < 1.*u.ns
        assert header.gps_time == header.time
        assert abs(header.time -
                   Time('2013-07-25T19:32:51.5492352')) < 1.*u.ns
        assert header.mutable is False
        with pytest.raises(TypeError):
            header['sub_int'] = 0

        with io.StringIO() as s:
            header.tofile(s)
            s.seek(0)
            assert s.readline().strip() == h_raw
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

    def test_rawdump_header_seek_offset(self):
        header = gsb.GSBHeader(tuple(self.rawdump_ts.split()))
        header_size = len(self.rawdump_ts) + 1
        assert header.seek_offset(1) == header_size
        assert header.seek_offset(10) == 10 * header_size

    def test_phased_header_seek_offset(self):
        fh = gsb.open(SAMPLE_HEADER, 'rt')
        header1 = gsb.GSBHeader.fromfile(fh, verify=True)
        fh.seek(0)
        header_size = len(fh.readline())
        # Sample file contains trailing white space
        assert header_size - header1.seek_offset(1) == 1
        # Check that seek_offset is working properly
        n_1000_0 = 1000 - 9980
        offset_to_1000_0 = header1.seek_offset(n_1000_0, size=header_size)
        assert offset_to_1000_0 == n_1000_0 * header_size
        # Go to 1000
        assert (header1.seek_offset(n_1000_0 - 1, size=header_size) ==
                (n_1000_0 - 1) * header_size + 1)
        # Go to 100 (header decreases by 1 chr)
        assert (header1.seek_offset(n_1000_0 - 900, size=header_size) ==
                (n_1000_0 - 900) * header_size + 900)
        # Go to 99 (header decreases by 2 chr)
        assert (header1.seek_offset(n_1000_0 - 901, size=header_size) ==
                (n_1000_0 - 901) * header_size + 902)
        # Go to 100001
        assert (header1.seek_offset(n_1000_0 + 99001, size=header_size) ==
                (n_1000_0 + 99001) * header_size + 90002)

        # Try retrieving headers using seek_offset
        fh.seek(header1.seek_offset(10, size=header_size))
        header2 = gsb.GSBHeader.fromfile(fh, verify=True)
        fh.seek(10 * header_size)
        h2_raw = fh.readline().strip()
        assert header2.mode == 'phased'
        assert header2['pc'] == h2_raw[:28]
        assert header2['gps'] == h2_raw[29:60]
        assert header2['seq_nr'] == 9990
        assert header2['sub_int'] == 6
        assert abs(header2.pc_time -
                   Time('2013-07-25T19:32:54.250583')) < 1.*u.ns
        assert header2.gps_time == header2.time
        assert abs(header2.time -
                   Time('2013-07-25T19:32:54.0658176')) < 1.*u.ns

        # Retrieve beyond sequence number 10000
        fh.seek(header1.seek_offset(33, size=header_size))
        header3 = gsb.GSBHeader.fromfile(fh, verify=True)
        fh.seek(33 * header_size + 13)
        h3_raw = fh.readline().strip()
        assert header3.mode == 'phased'
        assert header3['pc'] == h3_raw[:28]
        assert header3['gps'] == h3_raw[29:60]
        assert header3['seq_nr'] == 10013
        assert header3['sub_int'] == 5
        assert abs(header3.pc_time -
                   Time('2013-07-25T19:33:00.038495')) < 1.*u.ns
        assert header3.gps_time == header3.time
        assert abs(header3.time -
                   Time('2013-07-25T19:32:59.85395712')) < 1.*u.ns

        fh.close()

    def test_header_non_gmrt(self):
        # Assume file is non-GMRT
        with gsb.open(SAMPLE_HEADER, 'rt') as fh:
            header = gsb.GSBHeader.fromfile(fh, verify=True,
                                            utc_offset=0.*u.hr)
        assert abs(header.pc_time -
                   Time('2013-07-26T01:02:51.733965')) < 1.*u.ns
        assert header.gps_time == header.time
        assert abs(header.time -
                   Time('2013-07-26T01:02:51.5492352')) < 1.*u.ns

    def test_decoding(self):
        """Check that 4-bit encoding works."""
        areal = np.arange(-8., 8.)
        b = encode_4bit(areal)
        assert np.all(b.view(np.uint8) == np.array(
            [0x98, 0xba, 0xdc, 0xfe, 0x10, 0x32, 0x54, 0x76]))
        d = decode_4bit(b)
        assert np.all(d == areal)
        # Also check shape is preserved.
        areal2 = np.hstack((areal, areal[::-1]))
        b2 = encode_4bit(areal2)
        assert np.all(b2.view(np.uint8) == np.array(
            [0x98, 0xba, 0xdc, 0xfe, 0x10, 0x32, 0x54, 0x76,
             0x67, 0x45, 0x23, 0x01, 0xef, 0xcd, 0xab, 0x89]))
        d2 = decode_4bit(b2)
        assert np.all(d2 == areal2)

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
        assert np.all(payload6.words == payload1.words)
        assert payload6 == payload1

    @pytest.mark.parametrize('bps', (4, 8))
    def test_phased_payload_minimal(self, bps):
        payload = gsb.GSBPayload.fromdata(self.data[:1024*8//bps], bps=bps)
        # create a "fake" phased payload by writing the same data to files
        with io.BytesIO() as s1, io.BytesIO() as s2:
            payload.tofile(s1)
            payload.tofile(s2)
            s1.seek(0)
            s2.seek(0)
            if bps == 4:
                with pytest.raises(TypeError):
                    gsb.GSBPayload.fromfile([[s1], [s2]], bps=bps,
                                            payloadsize=payload.size)
            else:
                phased = gsb.GSBPayload.fromfile([[s1], [s2]], bps=bps,
                                                 payloadsize=payload.size)
                assert np.all(phased.data == payload.data[:, np.newaxis])

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
        data = self.data.reshape(-1, 1, 2)
        with gsb.open(SAMPLE_HEADER, 'rt') as fh:
            header1 = gsb.GSBHeader.fromfile(fh, verify=True)
        frame1 = gsb.GSBFrame.fromdata(data, header1, bps=4)
        assert frame1.shape == data.shape
        assert np.all(frame1.data == data)
        assert frame1.size == data.size // 2
        cmplx = data[::2] + 1j * data[1::2]
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
                                           payloadsize=data.size // 4,
                                           nchan=2)
        assert frame3 == frame1
        # Two polarisations, 16 channels
        twopol = cmplx.reshape(-1, 2, 16)
        frame4 = gsb.GSBFrame.fromdata(twopol, header1, bps=4)
        assert frame4.size == frame1.size  # number of bytes doesn't change.
        assert frame4.shape == twopol.shape
        assert np.all(frame4.data == twopol)
        with io.StringIO() as sh, \
                io.BytesIO() as sp0, io.BytesIO() as sp1, \
                io.BytesIO() as sp2, io.BytesIO() as sp3:
            frame4.tofile(sh, ((sp0, sp1), (sp2, sp3)))
            for s in sp0, sp1, sp2, sp3:
                assert s.tell() == frame4.payload.size // 4
                s.flush()
                s.seek(0)
            sh.flush()
            sh.seek(0)
            frame5 = gsb.GSBFrame.fromfile(sh, ((sp0, sp1), (sp2, sp3)),
                                           bps=4, payloadsize=data.size // 8,
                                           complex_data=True, nchan=16)
        assert frame5 == frame4

    def test_timestamp_io(self, tmpdir):
        with gsb.open(SAMPLE_HEADER, 'rt') as fh:
            header = gsb.GSBHeader.fromfile(fh, verify=True)
        tmpfile = str(tmpdir.join('timestamps.txt'))
        with gsb.open(tmpfile, 'wt') as fh_w:
            fh_w.write_timestamp(header)
            fh_w.write_timestamp(**header)

        with gsb.open(tmpfile, 'rt') as fh_r:
            for i in range(2):
                header2 = fh_r.read_timestamp()
                assert header2 == header

        # check that extra arguments raise TypeError
        with pytest.raises(TypeError):
            gsb.open(tmpfile, 'rt', raw='bla')

    def test_rawfile_io(self, tmpdir):
        payload = gsb.GSBPayload.fromdata(self.data, bps=4)
        tmpfile = str(tmpdir.join('payload.bin'))
        with gsb.open(tmpfile, 'wb') as fh_w:
            fh_w.write_payload(payload)
            fh_w.write_payload(payload.data, bps=payload.bps)

        with gsb.open(tmpfile, 'rb') as fh_r:
            for i in range(2):
                payload2 = fh_r.read_payload(
                    payload.size, bps=payload.bps,
                    nchan=payload.sample_shape[-1],
                    complex_data=payload.complex_data)
                assert payload2 == payload

        # check that extra arguments raise TypeError
        with pytest.raises(TypeError):
            gsb.open(tmpfile, 'rb', bps=2)

    def test_raw_stream(self, tmpdir):
        header = gsb.GSBHeader(self.rawdump_ts.split())
        payload = gsb.GSBPayload.fromdata(self.data, bps=4)
        # Test on files here, file handles below
        ts_file = str(tmpdir.join('ts.dat'))
        raw_file = str(tmpdir.join('raw0.dat'))
        with gsb.open(ts_file, 'wt') as sh:
            sh.write_timestamp(header)
        with io.open(raw_file, 'wb') as sp:
            payload.tofile(sp)

        # Open here with payloadsize given, below with samples_per_frame
        with gsb.open(ts_file, mode='rs', raw=raw_file,
                      payloadsize=payload.size, frames_per_second=1) as fh_r:
            assert fh_r.header0 == header
            data = fh_r.read()
            assert fh_r.tell() == len(data)
            assert_quantity_allclose(fh_r.tell(unit=u.s), 1. * u.s)
            assert fh_r.header1 == header
            # recheck with output array given
            out = np.zeros_like(self.data)
            fh_r.seek(0)
            fh_r.read(out=out)
        assert np.all(data == self.data.ravel())
        assert np.all(out == self.data)

        with io.BytesIO() as sh, io.BytesIO() as sp, gsb.open(
                sh, 'ws', raw=sp, sample_rate=4096*u.Hz,
                samples_per_frame=payload.nsample, **header) as fh_w:
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
                assert_quantity_allclose(
                    (fh_r.header1.time - fh_r.header0.time).to(u.s),
                    u.s/fh_r.frames_per_second)
            assert np.all(data.reshape(2, -1) == self.data.ravel())

        # Test that opening that raises an exception correctly handles
        # file closing. (Note that the timestamp file always gets closed).
        with io.BytesIO() as sh, io.BytesIO() as sp:
            with pytest.raises(u.UnitsError):
                gsb.open(sh, 'ws', raw=sp, sample_rate=4096*u.m,
                         samples_per_frame=payload.nsample, **header)
            assert not sp.closed

    @pytest.mark.parametrize('bps', (4, 8))
    def test_phased_stream(self, bps, tmpdir):
        with gsb.open(SAMPLE_HEADER, 'rt') as fh:
            header = gsb.GSBHeader.fromfile(fh, verify=True)
        # Single polarisation
        cmplx = self.data[::2] + 1j * self.data[1::2]
        onepol = cmplx.reshape(-1, 1, 16)
        # Test on files here, file handles below
        ts_file = str(tmpdir.join('ts.dat'))
        raw0_file = str(tmpdir.join('raw0.dat'))
        raw1_file = str(tmpdir.join('raw1.dat'))
        with gsb.open(ts_file, 'ws', raw=(raw0_file, raw1_file),
                      bps=bps, sample_rate=256*u.Hz,
                      samples_per_frame=onepol.shape[0] // 2,
                      nchan=onepol.shape[2], nthread=1,
                      complex_data=True, header=header) as fh_w:
            # Write data twice.
            fh_w.write(onepol)
            fh_w.write(onepol[::-1])
            assert fh_w.tell() == onepol.shape[0] * 2
            assert_quantity_allclose(fh_w.tell(unit=u.s), 0.5 * u.s)
            assert fh_w._header['seq_nr'] == fh_w.header0['seq_nr'] + 3
            fh_w.flush()
            assert fh_w.fh_raw[0][0].tell() == 2048 * bps // 8

        with gsb.open(ts_file, mode='rs', raw=(raw0_file, raw1_file),
                      bps=bps, samples_per_frame=onepol.shape[0] // 2,
                      nchan=onepol.shape[2]) as fh_r:
            assert fh_r.header0 == header
            assert np.isclose(fh_r.frames_per_second, 8.)
            data = fh_r.read(onepol.shape[0] * 2)
            assert fh_r.tell() == onepol.shape[0] * 2
            assert (fh_r._frame.header['seq_nr'] ==
                    fh_r.header0['seq_nr'] + 3)
            assert_quantity_allclose(fh_r.tell(unit=u.s), 0.5 * u.s)
            assert fh_r.fh_raw[0][0].tell() == 2048 * bps // 8
            assert fh_r._frame.header == fh_r.header1

        assert np.all(data[:onepol.shape[0]] == onepol.squeeze())
        assert np.all(data[onepol.shape[0]:] == onepol[::-1].squeeze())
        # Two polarisations, 16 channels
        twopol = cmplx.reshape(-1, 2, 16)
        with io.BytesIO() as sh, \
                io.BytesIO() as sp0, io.BytesIO() as sp1, \
                io.BytesIO() as sp2, io.BytesIO() as sp3, \
                gsb.open(sh, 'ws', raw=((sp0, sp1), (sp2, sp3)),
                         bps=bps, sample_rate=128*u.Hz,
                         samples_per_frame=twopol.shape[0] // 2,
                         nchan=twopol.shape[2], nthread=twopol.shape[1],
                         complex_data=True, header=header) as fh_w:
            # Write data twice.
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
                assert (fh_r._frame.header['seq_nr'] ==
                        fh_r.header0['seq_nr'] + 3)
                assert_quantity_allclose(fh_r.tell(unit=u.s), 0.5 * u.s)
                assert sp0.tell() == 1024 * bps // 8
                assert fh_r._frame.header == fh_r.header1
            assert np.all(data[:twopol.shape[0]] == twopol)
            assert np.all(data[twopol.shape[0]:] == twopol[::-1])

    def test_stream_invalid(self, tmpdir):
        with pytest.raises(ValueError):
            # no r or w in mode
            gsb.open('ts.dat', 's')
        with pytest.raises(TypeError), io.TextIOBase() as s:
            # TextIOBase for fh
            gsb.open(s, 'rt')
        with pytest.raises(IOError):
            # non-existing file
            gsb.open(str(tmpdir.join('ts.bla')),
                     raw=str(tmpdir.join('raw.bla')))
