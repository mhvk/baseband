# Licensed under the GPLv3 - see LICENSE
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.tests.helper import catch_warnings
from ... import gsb
from ..payload import decode_4bit, encode_4bit
from ...data import (SAMPLE_GSB_RAWDUMP_HEADER as SAMPLE_RAWDUMP_HEADER,
                     SAMPLE_GSB_RAWDUMP as SAMPLE_RAWDUMP,
                     SAMPLE_GSB_PHASED_HEADER as SAMPLE_PHASED_HEADER,
                     SAMPLE_GSB_PHASED as SAMPLE_PHASED)
from astropy.extern import six
if six.PY2:
    from io import open


class TestGSB(object):

    def setup(self):
        # For all sample files, each frame spans 0.25165824 sec.
        self.frame_rate = (1e8 / 3) / 2**23 * u.Hz
        # Payload size for all sample files is 2**12 bytes.
        self.payload_nbytes = 2**12

    def test_rawdump_header(self):
        with open(SAMPLE_RAWDUMP_HEADER, 'rt') as fh:
            header = gsb.GSBHeader.fromfile(fh, verify=True)
        assert header.mode == 'rawdump'
        assert header['gps'] == '2015 04 27 18 45 00 0.000000240'
        # Includes UTC offset.
        assert abs(header.time -
                   Time('2015-04-27T13:15:00.000000240')) < 1.*u.ns
        header2 = gsb.GSBHeader.fromkeys(**header)
        assert header2 == header
        header3 = gsb.GSBHeader.fromvalues(mode='rawdump', **header2)
        assert header3 == header2
        with pytest.raises(TypeError):
            gsb.GSBHeader.fromvalues(**header)
        with pytest.raises(TypeError):
            gsb.GSBHeader(None)
        # Check that recovering with the actual header type works as well.
        header4 = type(header).fromkeys(**header)
        assert header4 == header
        # Check that trying to initialize a phased header doesn't work.
        with pytest.raises(KeyError):
            gsb.header.GSBPhasedHeader.fromkeys(**header)
        header5 = header.copy()
        assert header5 == header

    def test_phased_header(self, tmpdir):
        with open(SAMPLE_PHASED_HEADER, 'rt') as fh:
            header = gsb.GSBHeader.fromfile(fh, verify=True)
            fh.seek(0)
            h_raw = fh.readline().strip()
        assert header.mode == 'phased'
        assert header['pc'] == h_raw[:28]
        assert header['gps'] == h_raw[29:60]
        assert header['seq_nr'] == 9995
        assert header['mem_block'] == 3
        assert abs(header.pc_time -
                   Time('2013-07-27T21:23:55.517535')) < 1.*u.ns
        assert header.gps_time == header.time
        assert abs(header.time -
                   Time('2013-07-27T21:23:55.3241088')) < 1.*u.ns
        assert header.mutable is False
        with pytest.raises(TypeError):
            header['mem_block'] = 0

        with open(str(tmpdir.join('test.timestamp')), 'w+t') as s:
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
                                           mem_block=header['mem_block'])
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

    def test_rawdump_header_seek_offset(self):
        fh = open(SAMPLE_RAWDUMP_HEADER, 'rt')

        header = gsb.GSBHeader.fromfile(fh, verify=True)
        # Includes 1 trailing blank space, one carriage return.
        header_nbytes = header.nbytes
        assert header_nbytes == len(' '.join(header.words)) + 2
        assert header.seek_offset(1) == header_nbytes
        assert header.seek_offset(12) == 12 * header_nbytes

        # Note that text pointers can't seek from current position.
        # Seek 2nd header.
        fh.seek(header.seek_offset(1))
        header1 = gsb.GSBHeader.fromfile(fh, verify=True)
        assert abs(header1.time -
                   Time('2015-04-27T13:15:00.251658480')) < 1.*u.ns

        fh.seek(header.seek_offset(9))
        header2 = gsb.GSBHeader.fromfile(fh, verify=True)
        assert abs(header2.time -
                   Time('2015-04-27T13:15:02.264924400')) < 1.*u.ns

        fh.close()

    def test_phased_header_seek_offset(self):
        fh = open(SAMPLE_PHASED_HEADER, 'rt')
        header1 = gsb.GSBHeader.fromfile(fh, verify=True)
        fh.seek(0)
        header_nbytes = header1.nbytes
        assert header_nbytes == header1.seek_offset(1)
        assert header_nbytes == len(fh.readline())
        # Check that seek_offset is working properly.
        n_1000_0 = 1000 - 9995
        offset_to_1000_0 = header1.seek_offset(n_1000_0)
        assert offset_to_1000_0 == n_1000_0 * header_nbytes
        # Go to 1000.
        assert (header1.seek_offset(n_1000_0 - 1) ==
                (n_1000_0 - 1) * header_nbytes + 1)
        # Go to 100 (header decreases by 1 chr).
        assert (header1.seek_offset(n_1000_0 - 900) ==
                (n_1000_0 - 900) * header_nbytes + 900)
        # Go to 99 (header decreases by 2 chr).
        assert (header1.seek_offset(n_1000_0 - 901) ==
                (n_1000_0 - 901) * header_nbytes + 902)
        # Go to 100001.
        assert (header1.seek_offset(n_1000_0 + 99001) ==
                (n_1000_0 + 99001) * header_nbytes + 90002)

        # Try retrieving headers using seek_offset.
        fh.seek(header1.seek_offset(3))
        header2 = gsb.GSBHeader.fromfile(fh, verify=True)
        fh.seek(3 * header_nbytes)
        h2_raw = fh.readline().strip()
        assert header2.mode == 'phased'
        assert header2['pc'] == h2_raw[:28]
        assert header2['gps'] == h2_raw[29:60]
        assert header2['seq_nr'] == 9998
        assert header2['mem_block'] == 6
        assert abs(header2.pc_time -
                   Time('2013-07-27T21:23:56.272643')) < 1.*u.ns
        assert header2.gps_time == header2.time
        assert abs(header2.time -
                   Time('2013-07-27T21:23:56.079083520')) < 1.*u.ns

        # Retrieve beyond sequence number 10000.
        fh.seek(header1.seek_offset(8))
        header3 = gsb.GSBHeader.fromfile(fh, verify=True)
        fh.seek(8 * header_nbytes + 3)
        h3_raw = fh.readline().strip()
        assert header3.mode == 'phased'
        assert header3['pc'] == h3_raw[:28]
        assert header3['gps'] == h3_raw[29:60]
        assert header3['seq_nr'] == 10003
        assert header3['mem_block'] == 3
        assert abs(header3.pc_time -
                   Time('2013-07-27T21:23:57.530805')) < 1.*u.ns
        assert header3.gps_time == header3.time
        assert abs(header3.time -
                   Time('2013-07-27T21:23:57.337374720')) < 1.*u.ns

        fh.close()

    def test_header_non_gmrt(self):
        # Assume file is non-GMRT.
        with open(SAMPLE_PHASED_HEADER, 'rt') as fh:
            header = gsb.GSBHeader.fromfile(fh, verify=True,
                                            utc_offset=0.*u.hr)
        assert abs(header.pc_time -
                   Time('2013-07-28T02:53:55.517535')) < 1.*u.ns
        assert header.gps_time == header.time
        assert abs(header.time -
                   Time('2013-07-28T02:53:55.3241088')) < 1.*u.ns

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
        with open(SAMPLE_RAWDUMP, 'rb') as fh:
            # For rawdump, payload_nbytes = (
            #   8192 samples/frame * 0.5 bytes/sample)
            payload1 = gsb.GSBPayload.fromfile(
                fh, payload_nbytes=self.payload_nbytes)
            # Values from reading original raw file.
            assert np.all(payload1.data[:20].ravel() == np.array(
                [0., -2., -2., 0., 4., -1., -2., -1., 1., 2., -1., 1.,
                 -1., 1., -2., 0., -1., -2., 1., -1.], dtype=np.float32))
            assert payload1.data.shape == (8192, 1)
            assert payload1.sample_shape == (1,)
            assert payload1.sample_shape.nchan == 1
            assert payload1.shape == (8192, 1)
            assert payload1.size == 8192
            assert payload1.ndim == 2
            with pytest.raises(ValueError):
                gsb.GSBPayload.fromfile(fh, payload_nbytes=None)
            payload2 = gsb.GSBPayload.fromdata(payload1.data, bps=4)
            assert np.all(payload2.data == payload1.data)
            payload3 = gsb.GSBPayload(payload1.words, bps=4,
                                      sample_shape=payload1.sample_shape)
            assert np.all(payload3.data == payload1.data)

        with open(SAMPLE_PHASED[0][0], 'rb') as fh:
            # For 1 file in phased, payload_nbytes = (
            #   4 full samples/file * 512 channels/full sample *
            #   2 complex elements/channel * 1 bytes/element)
            payload4 = gsb.GSBPayload.fromfile(
                fh, bps=8, complex_data=True,
                payload_nbytes=self.payload_nbytes)
            assert np.all(payload4.data[:20].ravel() == np.array(
                [30. + 12.j, -1. + 8.j, 7. + 19.j, -25. - 5.j, 26. + 14.j,
                 -9. + 0.j, -4. - 1.j, 7. + 6.j, 3. + 5.j, 1. - 2.j,
                 1. - 5.j, 10. - 6.j, 15. - 11.j, -6. + 13.j, 7. + 0.j,
                 -10. - 1.j, -8. + 7.j, 13. + 7.j, -1. + 1.j, 0. + 4.j],
                dtype=np.complex64))
            assert payload4.data.shape == (2048, 1)
            assert payload4.sample_shape == (1,)
            assert payload4.sample_shape.nchan == 1
            payload5 = gsb.GSBPayload.fromdata(payload4.data, bps=8)
            assert np.all(payload5.words == payload4.words)
            payload6 = gsb.GSBPayload(payload4.words, bps=8, complex_data=True,
                                      sample_shape=payload4.sample_shape)
            assert np.all(payload6.data == payload4.data)

        # Test encoding then decoding consistency via `fromdata`.
        payload7 = gsb.payload.GSBPayload.fromdata(payload4.data.real, bps=8)
        assert np.all(payload7.data == payload4.data.real)
        # Encode complex data.
        payload8 = gsb.GSBPayload.fromdata(payload4.data, bps=8)
        assert np.all(payload8.data == payload4.data)
        assert np.all(payload8.words == payload8.words)
        # Encode channelized data.
        channelized = payload4.data.reshape(-1, 512)
        payload9 = gsb.GSBPayload.fromdata(channelized, bps=8)
        assert payload9.shape == channelized.shape
        assert payload9.sample_shape == (512,)
        assert payload9.sample_shape.nchan == 512
        assert np.all(payload9.words == payload4.words)

    def test_phased_payload(self):
        """Test passing tuple into GSBPayload.fromfile.

        (Check that the same result is obtained with a set of single files.)
        """
        # Open tuple of tuple of binary filehandles.
        fh = [[open(thread, 'rb') for thread in pol]
              for pol in SAMPLE_PHASED]
        phased = gsb.GSBPayload.fromfile(
            fh, payload_nbytes=self.payload_nbytes, nchan=512,
            bps=8, complex_data=True)
        assert phased.shape == (8, 2, 512)
        assert phased.sample_shape == (2, 512)
        # Open equivalent sequence of individual payloads, and extract data.
        idata = np.empty([2, 2, 2048], dtype=np.complex64)
        for i, pol in enumerate(SAMPLE_PHASED):
            for j, thread in enumerate(pol):
                with open(thread, 'rb') as ft:
                    ftpayload = gsb.GSBPayload.fromfile(
                        ft, payload_nbytes=self.payload_nbytes,
                        bps=8, complex_data=True)
                    idata[i, j] = ftpayload.data[:, 0]
        # Channelize and merge threads.
        idata = idata.reshape(2, 8, 512).transpose(1, 0, 2)
        assert np.all(phased.data == idata)
        # Raises error for incorrect bps * nchan.
        with pytest.raises(TypeError):
            gsb.GSBPayload.fromfile(fh, payload_nbytes=self.payload_nbytes,
                                    nchan=1, bps=4)
        # Close file handles.
        for pol in fh:
            for thread in pol:
                thread.close()

    def test_rawdump_frame(self, tmpdir):
        # Load rawdump frame.
        with open(SAMPLE_RAWDUMP_HEADER, 'rt') as ft, \
                open(SAMPLE_RAWDUMP, 'rb') as fraw:
            frame1 = gsb.GSBFrame.fromfile(ft, fraw, bps=4,
                                           payload_nbytes=self.payload_nbytes)
        # Open sections individually.
        with open(SAMPLE_RAWDUMP_HEADER, 'rt') as fh:
            header1 = gsb.GSBHeader.fromfile(fh, verify=True)
        with open(SAMPLE_RAWDUMP, 'rb') as fh:
            payload1 = gsb.GSBPayload.fromfile(
                fh, payload_nbytes=self.payload_nbytes)
        # Compare the two.
        assert header1 == frame1.header
        assert np.all(payload1.data == frame1.payload.data)
        assert frame1.shape == payload1.shape
        assert frame1.size == payload1.size
        assert frame1.ndim == payload1.ndim

        frame2 = gsb.GSBFrame(frame1.header, frame1.payload)
        assert frame2 == frame1
        with open(str(tmpdir.join('test.timestamp')), 'w+t') as sh, \
                open(str(tmpdir.join('test.dat')), 'w+b') as sp:
            frame1.tofile(sh, sp)
            sh.seek(0)
            sp.seek(0)
            frame3 = gsb.GSBFrame.fromfile(sh, sp, bps=4,
                                           payload_nbytes=frame1.nbytes)
        assert frame3 == frame1

    def seek_phased_rawfiles(self, fraw, offset):
        for pol in fraw:
            for thread in pol:
                thread.seek(offset)

    def close_phased_rawfiles(self, fraw):
        for pol in fraw:
            for thread in pol:
                thread.close()

    def test_phased_frame(self, tmpdir):
        # Load phased frame.
        fraw = [[open(thread, 'rb') for thread in pol]
                for pol in SAMPLE_PHASED]
        with open(SAMPLE_PHASED_HEADER, 'rt') as ft:
            frame1 = gsb.GSBFrame.fromfile(ft, fraw, nchan=512, bps=8,
                                           payload_nbytes=self.payload_nbytes,
                                           complex_data=True)
        self.seek_phased_rawfiles(fraw, 0)
        # Open sections individually.
        with open(SAMPLE_PHASED_HEADER, 'rt') as fh:
            header1 = gsb.GSBHeader.fromfile(fh, verify=True)
        payload1 = gsb.GSBPayload.fromfile(
            fraw, payload_nbytes=self.payload_nbytes, nchan=512, bps=8,
            complex_data=True)
        # Compare the two.
        assert frame1.dtype.kind == 'c'
        assert header1 == frame1.header
        assert frame1.shape == payload1.shape
        assert frame1.size == payload1.size
        assert frame1.ndim == payload1.ndim
        assert np.all(frame1.payload.data == payload1.data)
        assert frame1.valid is True
        frame1.valid = False
        assert frame1.valid is False
        assert np.all(frame1.data == 0.)
        frame1.valid = True
        assert frame1.valid is True
        assert np.all(frame1.payload.data == payload1.data)
        self.close_phased_rawfiles(fraw)

        # Try only right polarization.
        fraw = [[open(thread, 'rb') for thread in SAMPLE_PHASED[1]]]
        with open(SAMPLE_PHASED_HEADER, 'rt') as ft:
            frame2 = gsb.GSBFrame.fromfile(
                ft, fraw, payload_nbytes=self.payload_nbytes, nchan=512, bps=8,
                complex_data=True)
        self.seek_phased_rawfiles(fraw, 0)
        payload2 = gsb.GSBPayload.fromfile(
            fraw, payload_nbytes=self.payload_nbytes, nchan=512, bps=8,
            complex_data=True)
        assert frame2.shape == payload2.shape
        assert np.all(frame2.payload.data == payload2.data)
        self.close_phased_rawfiles(fraw)

        frame3 = gsb.GSBFrame.fromdata(payload1.data, header1, bps=8)
        assert frame3.shape == frame1.shape
        assert np.all(frame3.data == frame1.data)
        # Test writing phased data to multiple files.
        with open(str(tmpdir.join('test.timestamp')), 'w+t') as sh, \
                open(str(tmpdir.join('test0.dat')), 'w+b') as sp0, \
                open(str(tmpdir.join('test1.dat')), 'w+b') as sp1, \
                open(str(tmpdir.join('test2.dat')), 'w+b') as sp2, \
                open(str(tmpdir.join('test3.dat')), 'w+b') as sp3:
            frame1.tofile(sh, ((sp0, sp1), (sp2, sp3)))
            for s in sp0, sp1, sp2, sp3:
                s.flush()
                s.seek(0)
            sh.flush()
            sh.seek(0)
            frame4 = gsb.GSBFrame.fromfile(sh, ((sp0, sp1), (sp2, sp3)),
                                           payload_nbytes=self.payload_nbytes,
                                           nchan=512, bps=8, complex_data=True)
        assert frame4 == frame1

    @pytest.mark.parametrize('sample', (SAMPLE_RAWDUMP_HEADER,
                                        SAMPLE_PHASED_HEADER))
    def test_timestamp_io(self, tmpdir, sample):
        """Tests GSBTimeStampIO in base.py."""
        with open(sample, 'rt') as fh:
            header0 = gsb.GSBHeader.fromfile(fh, verify=True)

        with gsb.open(sample, 'rt') as fh:
            header1 = fh.read_timestamp()
            assert header1 == header0
            current_pos = fh.tell()
            frame_rate = fh.get_frame_rate()
            assert abs(frame_rate - u.Hz / 0.251658240) < 1. * u.nHz
            assert fh.tell() == current_pos

        testfile = str(tmpdir.join('test.timestamp'))
        with gsb.open(testfile, 'wt') as fw:
            fw.write_timestamp(header=header1)
        with gsb.open(testfile, 'rt') as fh:
            header2 = fh.read_timestamp()
            assert header2 == header1

        # Check that extra arguments raise TypeError.
        with pytest.raises(TypeError):
            gsb.open(testfile, 'rt', raw='bla')

    def test_rawfile_io(self, tmpdir):
        """Tests GSBFileReader and GSBFileWriter in base.py."""
        with open(SAMPLE_RAWDUMP, 'rb') as fh:
            payload1 = gsb.GSBPayload.fromfile(
                fh, payload_nbytes=self.payload_nbytes)
        testfile = str(tmpdir.join('test.dat'))
        with gsb.open(testfile, 'wb') as fw:
            fw.write_payload(payload1, bps=4)
        with gsb.open(testfile, 'rb', payload_nbytes=2**12) as fh:
            payload2 = fh.read_payload()
            assert payload2 == payload1

    def test_rawfile_repr(self):
        with gsb.open(SAMPLE_RAWDUMP, 'rb', payload_nbytes=2**12, nchan=1,
                      bps=4, complex_data=False) as fh:
            repr_fh = repr(fh)
        assert repr_fh.startswith('GSBFileReader')
        assert ('payload_nbytes=4096, nchan=1, bps=4, complex_data=False' in
                repr_fh)

    def test_raw_stream(self, tmpdir):
        bps = 4
        sample_rate = self.frame_rate * self.payload_nbytes * (8 // bps)
        # Open here with payloadsize given, below with samples_per_frame.
        with gsb.open(SAMPLE_RAWDUMP_HEADER, 'rs', raw=SAMPLE_RAWDUMP,
                      sample_rate=sample_rate,
                      payload_nbytes=self.payload_nbytes,
                      squeeze=False) as fh_r:
            with open(SAMPLE_RAWDUMP_HEADER, 'rt') as ft, \
                    open(SAMPLE_RAWDUMP, 'rb') as fraw:
                frame1 = gsb.GSBFrame.fromfile(
                    ft, fraw, bps=4, payload_nbytes=self.payload_nbytes)
            assert fh_r.header0.time == fh_r.start_time
            assert fh_r.header0 == frame1.header
            assert fh_r.sample_shape == (1,)
            assert fh_r.shape == ((10 * fh_r.samples_per_frame,) +
                                  fh_r.sample_shape)
            assert fh_r.size == np.prod(fh_r.shape)
            assert fh_r.ndim == len(fh_r.shape)
            assert fh_r.sample_rate == sample_rate
            check = fh_r.read(fh_r.samples_per_frame)
            assert np.all(check == frame1.data)
            # Seek last offset.
            with open(SAMPLE_RAWDUMP_HEADER, 'rt') as ft, \
                    open(SAMPLE_RAWDUMP, 'rb') as fraw:
                ft.seek(frame1.header.seek_offset(9))
                fraw.seek(9 * fh_r._payload_nbytes)
                frame10 = gsb.GSBFrame.fromfile(
                    ft, fraw, bps=4, payload_nbytes=self.payload_nbytes)
            assert fh_r._last_header == frame10.header
            fh_r.seek(-10, 2)
            check = fh_r.read(10)
            assert np.all(check == frame10.data[-10:])
            # Check validity of current and stopping time
            assert abs(fh_r.stop_time -
                       Time('2015-04-27T13:15:02.516582640')) < 1.*u.ns
            assert abs(fh_r.stop_time - fh_r.time) < 1.*u.ns
            fh_r.seek(0)
            data1 = fh_r.read()
            assert fh_r.tell() == len(data1)
            assert data1.shape == fh_r.shape
            fh_r.seek(0)
            out1 = np.empty_like(data1)
            fh_r.read(out=out1)
            assert np.all(out1 == data1)

        # Try with squeezing on.
        with gsb.open(SAMPLE_RAWDUMP_HEADER, 'rs', raw=SAMPLE_RAWDUMP,
                      sample_rate=sample_rate,
                      payload_nbytes=self.payload_nbytes,
                      squeeze=True) as fh_r:
            data2 = fh_r.read()
            assert np.all(data2 == data1.squeeze())
            out2 = np.empty(fh_r.shape)
            fh_r.seek(0)
            fh_r.read(out=out2)
            assert np.all(out2 == data1.squeeze())
            # To compare with directly psasing samples_per_frame below.
            spf_from_payload_nbytes = fh_r.samples_per_frame
            # For testing proper file closing below.
            header0 = fh_r.header0

        gsbtest_ts = str(tmpdir.join('test_time.timestamp'))
        gsbtest_raw = str(tmpdir.join('test.dat'))
        with gsb.open(SAMPLE_RAWDUMP_HEADER, 'rs', raw=SAMPLE_RAWDUMP,
                      sample_rate=sample_rate, samples_per_frame=(
                          self.payload_nbytes * (8 // bps))) as fh_r:
            # Check that passing samples_per_frame is identical to passing
            # payload_nbytes.
            assert fh_r.samples_per_frame == spf_from_payload_nbytes
            check = fh_r.read()
            assert np.all(check == data2)

            with gsb.open(gsbtest_ts, 'ws', raw=gsbtest_raw,
                          header0=fh_r.header0, sample_rate=fh_r.sample_rate,
                          samples_per_frame=(self.payload_nbytes * (8 // bps))
                          ) as fh_w:
                assert fh_w.sample_rate == sample_rate
                fh_w.write(data2)

            with gsb.open(gsbtest_ts, 'rs', raw=gsbtest_raw,
                          sample_rate=fh_r.sample_rate,
                          samples_per_frame=(
                              self.payload_nbytes * (8 // bps))) as fh_n:
                assert fh_n.header0 == fh_r.header0
                assert fh_n._last_header == fh_r._last_header
                assert fh_n.sample_shape == fh_r.sample_shape
                assert fh_n.shape == fh_r.shape
                assert fh_n.start_time == fh_r.start_time
                assert fh_n.sample_rate == sample_rate
                check = fh_n.read()
                assert np.all(check == data2)
                assert abs(fh_n.stop_time - fh_n.time) < 1.*u.ns
                assert abs(fh_n.stop_time - fh_r.stop_time) < 1.*u.ns

        # Try writing a file with squeeze off, and reading it back with squeeze.
        with gsb.open(gsbtest_ts, 'ws', raw=gsbtest_raw,
                      header0=header0, sample_rate=sample_rate,
                      samples_per_frame=(self.payload_nbytes * (8 // bps)),
                      squeeze=False) as fh_wns:
            fh_wns.write(data1)

        with gsb.open(gsbtest_ts, 'rs', raw=gsbtest_raw,
                      sample_rate=sample_rate,
                      samples_per_frame=(self.payload_nbytes *
                                         (8 // bps))) as fh_nns:
            check == fh_nns.read()
            assert np.all(check == data2)

        # Test that opening that raises an exception correctly handles
        # file closing. (Note that the timestamp file always gets closed).
        with open(str(tmpdir.join('test.timestamp')), 'w+b') as sh, \
                open(str(tmpdir.join('test.dat')), 'w+b') as sp:
            with pytest.raises(u.UnitsError):
                gsb.open(sh, 'ws', raw=sp, sample_rate=3.9736/u.m,
                         samples_per_frame=(self.payload_nbytes * (8 // bps)),
                         **header0)
            assert not sp.closed

        # Test that an incomplete last header leads to the second-to-last
        # header being used, and raises a warning.
        filename_incompletehead = str(
            tmpdir.join('test_incomplete_header.timestamp'))
        with open(SAMPLE_RAWDUMP_HEADER, 'rt') as fh, \
                open(filename_incompletehead, 'wt') as fw:
            fw.write(fh.read()[:-4])
        with gsb.open(filename_incompletehead, 'rs', raw=SAMPLE_RAWDUMP,
                      sample_rate=sample_rate,
                      payload_nbytes=self.payload_nbytes,
                      squeeze=False) as fh_r:
            with catch_warnings(UserWarning) as w:
                fh_r._last_header
            assert len(w) == 1
            assert 'second-to-last entry' in str(w[0].message)
            assert fh_r.shape[0] == 9 * fh_r.samples_per_frame
        with open(SAMPLE_RAWDUMP_HEADER, 'rt') as fh, \
                open(filename_incompletehead, 'wt') as fw:
            fw.write(fh.read()[:45])
        with gsb.open(filename_incompletehead, 'rs', raw=SAMPLE_RAWDUMP,
                      sample_rate=sample_rate,
                      payload_nbytes=self.payload_nbytes,
                      squeeze=False) as fh_r:
            with catch_warnings(UserWarning) as w:
                fh_r._last_header
            assert len(w) == 1
            assert 'second-to-last entry' in str(w[0].message)
            assert fh_r.shape[0] == fh_r.samples_per_frame
            assert fh_r._last_header == fh_r.header0

        # Test not passing a sample rate and samples per frame to reader
        # (can't test reading, since the sample file is tiny).
        with gsb.open(SAMPLE_RAWDUMP_HEADER, 'rs', raw=SAMPLE_RAWDUMP) as fh_r:
            assert fh_r.sample_rate == (100. / 3.) * u.MHz
            assert fh_r.samples_per_frame == 2**23
            assert fh_r._payload_nbytes == 2**22

    def test_phased_stream(self, tmpdir):
        bps = 8
        nchan = 512
        sample_rate = (self.frame_rate * self.payload_nbytes *
                       (8 // bps) / nchan)
        # Open here with payloadsize given, below with samples_per_frame.
        with gsb.open(SAMPLE_PHASED_HEADER, 'rs', raw=SAMPLE_PHASED,
                      sample_rate=sample_rate,
                      payload_nbytes=self.payload_nbytes,
                      squeeze=False) as fh_r:
            fraw = [[open(thread, 'rb') for thread in pol]
                    for pol in SAMPLE_PHASED]
            with open(SAMPLE_PHASED_HEADER, 'rt') as ft:
                frame1 = gsb.GSBFrame.fromfile(
                    ft, fraw, payload_nbytes=self.payload_nbytes, nchan=nchan,
                    bps=bps, complex_data=True)
            assert fh_r.header0.time == fh_r.start_time
            assert fh_r.header0 == frame1.header
            assert fh_r.sample_shape == (2, 512)
            assert fh_r.shape == ((10 * fh_r.samples_per_frame,) +
                                  fh_r.sample_shape)
            assert fh_r.size == np.prod(fh_r.shape)
            assert fh_r.ndim == len(fh_r.shape)
            assert fh_r.sample_rate == sample_rate
            assert np.all(fh_r.read(fh_r.samples_per_frame) == frame1.data)
            # Seek last offset.
            with open(SAMPLE_PHASED_HEADER, 'rt') as ft:
                ft.seek(frame1.header.seek_offset(9))
                self.seek_phased_rawfiles(fraw, 9 * fh_r._payload_nbytes)
                frame10 = gsb.GSBFrame.fromfile(
                    ft, fraw, payload_nbytes=self.payload_nbytes, nchan=nchan,
                    bps=bps, complex_data=True)
            assert fh_r._last_header == frame10.header
            fh_r.seek(-8, 2)
            assert np.all(fh_r.read(8) == frame10.data)
            # Check validity of current and stopping time.
            assert abs(fh_r.stop_time -
                       Time('2013-07-27T21:23:57.8406912')) < 1.*u.ns
            assert abs(fh_r.stop_time - fh_r.time) < 1.*u.ns
            fh_r.seek(0)
            data1 = fh_r.read()
            assert fh_r.tell() == len(data1)
            assert data1.shape == fh_r.shape
            fh_r.seek(0)
            out1 = np.empty_like(data1)
            fh_r.read(out=out1)
            assert np.all(out1 == data1)
            fh_r.seek(1, 'end')
            with pytest.raises(EOFError):
                fh_r.read()

        # Try again with squeezing.
        with gsb.open(SAMPLE_PHASED_HEADER, 'rs', raw=SAMPLE_PHASED,
                      sample_rate=sample_rate,
                      payload_nbytes=self.payload_nbytes,
                      squeeze=True) as fh_r:
            out2 = np.empty(fh_r.shape, dtype=np.complex64)
            fh_r.read(out=out2)
            assert np.all(out2 == out1.squeeze())
            # To compare with directly psasing samples_per_frame below.
            spf_from_payload_nbytes = fh_r.samples_per_frame
            self.close_phased_rawfiles(fraw)

        # Try only right polarization.
        with gsb.open(SAMPLE_PHASED_HEADER, 'rs', raw=SAMPLE_PHASED[1],
                      sample_rate=sample_rate,
                      payload_nbytes=self.payload_nbytes,
                      squeeze=False) as fh_r:

            fraw = [[open(thread, 'rb') for thread in SAMPLE_PHASED[1]]]
            with open(SAMPLE_PHASED_HEADER, 'rt') as ft:
                frame1 = gsb.GSBFrame.fromfile(
                    ft, fraw, payload_nbytes=self.payload_nbytes, nchan=nchan,
                    bps=bps, complex_data=True)
            assert fh_r.header0.time == fh_r.start_time
            assert fh_r.header0 == frame1.header
            assert np.all(fh_r.read(fh_r.samples_per_frame) == frame1.data)
            self.close_phased_rawfiles(fraw)

        # Test subsetting.
        with gsb.open(SAMPLE_PHASED_HEADER, 'rs', raw=SAMPLE_PHASED,
                      sample_rate=sample_rate, subset=(1, 3),
                      payload_nbytes=self.payload_nbytes) as fh_r:
            assert fh_r.sample_shape == ()
            assert np.all(fh_r.read() == data1[:, 1, 3])

        subset_md = (np.array([1, 0])[:, np.newaxis], [1, 33, 121, 245])
        with gsb.open(SAMPLE_PHASED_HEADER, 'rs', raw=SAMPLE_PHASED,
                      sample_rate=sample_rate,
                      payload_nbytes=self.payload_nbytes,
                      subset=subset_md) as fh_r:
            assert fh_r.sample_shape == (2, 4)
            assert np.all(fh_r.read() == data1[(slice(None),) + subset_md])

        with gsb.open(SAMPLE_PHASED_HEADER, 'rs', raw=SAMPLE_PHASED[1],
                      sample_rate=sample_rate,
                      payload_nbytes=self.payload_nbytes,
                      subset=slice(0, 256)) as fh_r:
            assert fh_r.sample_shape == (256,)
            fraw = [[open(thread, 'rb') for thread in SAMPLE_PHASED[1]]]
            with open(SAMPLE_PHASED_HEADER, 'rt') as ft:
                frame1 = gsb.GSBFrame.fromfile(
                    ft, fraw, payload_nbytes=self.payload_nbytes, nchan=nchan,
                    bps=bps, complex_data=True)
            assert np.all(fh_r.read(fh_r.samples_per_frame) ==
                          frame1.data[:, 0, :256])
            self.close_phased_rawfiles(fraw)

        # Try writing to file by passing header keywords into open.
        with gsb.open(SAMPLE_PHASED_HEADER, 'rs', raw=SAMPLE_PHASED,
                      sample_rate=sample_rate,
                      samples_per_frame=(self.payload_nbytes //
                                         nchan)) as fh_r, \
                open(str(tmpdir.join('test_time.timestamp')), 'w+b') as sh,\
                open(str(tmpdir.join('test0.dat')), 'w+b') as sp0, \
                open(str(tmpdir.join('test1.dat')), 'w+b') as sp1, \
                open(str(tmpdir.join('test2.dat')), 'w+b') as sp2, \
                open(str(tmpdir.join('test3.dat')), 'w+b') as sp3:
            fraw = ((sp0, sp1), (sp2, sp3))
            fh_w = gsb.open(sh, 'ws', raw=fraw, sample_rate=fh_r.sample_rate,
                            samples_per_frame=(self.payload_nbytes // nchan),
                            **fh_r.header0)
            assert fh_w.sample_rate == sample_rate
            fh_w.write(fh_r.read())
            fh_w.flush()
            sh.seek(0)
            self.seek_phased_rawfiles(fraw, 0)
            fh_r.seek(0)
            with gsb.open(sh, 'rs', raw=fraw, sample_rate=sample_rate,
                          samples_per_frame=(self.payload_nbytes // nchan)
                          ) as fh_n:
                assert fh_n.header0 == fh_r.header0
                # PC time will not be the same.
                for key in ('gps', 'seq_nr', 'mem_block'):
                    assert fh_n._last_header[key] == fh_r._last_header[key]
                assert fh_n.shape == fh_r.shape
                assert fh_n.sample_shape == fh_r.sample_shape
                assert fh_n.start_time == fh_r.start_time
                assert fh_n.sample_rate == sample_rate
                assert np.all(fh_n.read() == fh_r.read())
                assert abs(fh_n.stop_time - fh_n.time) < 1.*u.ns
                assert abs(fh_n.stop_time - fh_r.stop_time) < 1.*u.ns
            # Check that passing samples_per_frame is identical to passing
            # payload_nbytes.
            fh_r.seek(0)
            assert fh_r.samples_per_frame == spf_from_payload_nbytes
            assert np.all(fh_r.read() == data1)
            fh_w.close()

        # Test that an incomplete last header leads to the second-to-last
        # header being used, and raises a warning.
        filename_incompletehead = str(
            tmpdir.join('test_incomplete_header.timestamp'))
        with open(SAMPLE_PHASED_HEADER, 'rt') as fh, \
                open(filename_incompletehead, 'wt') as fw:
            fw.write(fh.read()[:-7])
        with gsb.open(filename_incompletehead, 'rs', raw=SAMPLE_PHASED,
                      sample_rate=sample_rate,
                      payload_nbytes=self.payload_nbytes,
                      squeeze=False) as fh_r:
            with catch_warnings(UserWarning) as w:
                fh_r._last_header
            assert len(w) == 1
            assert 'second-to-last entry' in str(w[0].message)
            assert fh_r.shape[0] == 9 * fh_r.samples_per_frame
        with open(SAMPLE_PHASED_HEADER, 'rt') as fh, \
                open(filename_incompletehead, 'wt') as fw:
            fw.write(fh.read()[:97])
        with gsb.open(filename_incompletehead, 'rs', raw=SAMPLE_PHASED,
                      sample_rate=sample_rate,
                      payload_nbytes=self.payload_nbytes,
                      squeeze=False) as fh_r:
            with catch_warnings(UserWarning) as w:
                fh_r._last_header
            assert len(w) == 1
            assert 'second-to-last entry' in str(w[0].message)
            assert fh_r.shape[0] == fh_r.samples_per_frame
            assert fh_r._last_header == fh_r.header0

        # Test not passing a sample rate and samples per frame to reader
        # (can't test reading, since the sample file is tiny).
        with gsb.open(SAMPLE_PHASED_HEADER, 'rs', raw=SAMPLE_PHASED) as fh_r:
            assert fh_r.sample_rate == (fh_r.samples_per_frame *
                                        (100. / 3. / 2.**23) * u.MHz)
            assert fh_r.samples_per_frame == 2**13  # 2**23 / 1024
            assert fh_r._payload_nbytes == 2**22

    def test_stream_invalid(self, tmpdir):
        with pytest.raises(ValueError):
            # no r or w in mode
            gsb.open('ts.dat', 's')
        with pytest.raises(TypeError), \
                open(str(tmpdir.join('test.timestamp')), 'w+t') as s:
            # TextIOBase for fh
            gsb.open(s, 'rt')
        with pytest.raises(IOError):
            # non-existing file
            gsb.open(str(tmpdir.join('ts.bla')),
                     raw=str(tmpdir.join('raw.bla')))
