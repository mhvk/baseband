# Licensed under the GPLv3 - see LICENSE
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pytest
import copy
import numpy as np
import astropy.units as u
from astropy.tests.helper import catch_warnings
from ... import guppi
from ...helpers import sequentialfile as sf
from ..base import GUPPIFileNameSequencer
from ...data import SAMPLE_PUPPI as SAMPLE_FILE


class TestGUPPI(object):
    def setup(self):
        with open(SAMPLE_FILE, 'rb') as fh:
            self.header = guppi.GUPPIHeader.fromfile(fh)
            self.payload = guppi.GUPPIPayload.fromfile(fh, self.header,
                                                       memmap=False)
        # Create a header with no overlap for stream writers.
        self.header_w = self.header.copy()
        self.header_w.overlap = 0
        self.header_w.payload_nbytes = self.header.payload_nbytes - (
            self.header._bpcs * self.header.overlap // 8)

    def test_header(self, tmpdir):
        with open(SAMPLE_FILE, 'rb') as fh:
            header = guppi.GUPPIHeader.fromfile(fh)
            assert header.nbytes == 6400
            assert fh.tell() == 6400
        assert header['OBSNCHAN'] == 4
        assert header['STT_IMJD'] == 58132
        assert header['STT_SMJD'] == 51093
        assert header['STT_OFFS'] == 0
        assert header['PKTIDX'] == 0
        assert header['PKTSIZE'] == 1024
        assert header.time.isot == '2018-01-14T14:11:33.000'
        assert header.payload_nbytes == 16384
        assert header.frame_nbytes == 16384 + 6400
        assert header.overlap == 64
        assert header.samples_per_frame == 1024
        assert header.mutable is False
        with pytest.raises(TypeError):
            header['OBSNCHAN'] = 213
        assert header['OBSNCHAN'] == 4
        with pytest.raises(AttributeError):
            header.python

        with open(str(tmpdir.join('testguppi.raw')), 'w+b') as s:
            header.tofile(s)
            assert s.tell() == header.nbytes
            s.seek(0)
            header2 = guppi.GUPPIHeader.fromfile(s)
            assert header2 == header
            assert header2.mutable is False
            assert s.tell() == header.nbytes

        # A header missing "END" raises a decoding error.
        with open(SAMPLE_FILE, 'rb') as fh, \
                open(str(tmpdir.join('testguppi.raw')), 'w+b') as s:
            # Create file without "END" in first header.
            s.write(fh.read(6320))
            fh.seek(6400)
            s.write(fh.read(10000))
            s.seek(0)
            # Try reading in file.
            with pytest.raises(UnicodeDecodeError):
                header = guppi.GUPPIHeader.fromfile(s)

        # Note that this is not guaranteed to preserve order!
        header3 = guppi.GUPPIHeader.fromkeys(**header)
        assert header3 == header
        assert header3.mutable is True
        # Check setting attributes.
        header3.start_time = header.start_time - 0.5 * u.day
        assert np.abs(header3.start_time -
                      (header.start_time - 0.5 * u.day)) < 1 * u.ns
        assert np.abs(header3.time - (header.time - 0.5 * u.day)) < 1 * u.ns
        header3.frame_nbytes = 13000
        assert header3.payload_nbytes == 6600
        header4 = guppi.GUPPIHeader.fromkeys(**header)
        packet_time = (header4['PKTSIZE'] * 8 // header4['OBSNCHAN'] //
                       header4['NPOL'] // header4.bps) * header4['TBIN'] * u.s
        header4.offset += packet_time
        assert header4['PKTIDX'] == header['PKTIDX'] + 1
        assert np.abs(header4.time -
                      (header.time + packet_time)) < 1 * u.ns

        header5 = guppi.GUPPIHeader.fromvalues(
            start_time=header.start_time,
            sample_rate=header.sample_rate,
            samples_per_frame=header.samples_per_frame,
            overlap=header.overlap, sample_shape=header.sample_shape,
            sideband=header.sideband, bps=header.bps,
            stt_offs=header['STT_OFFS'], pktidx=header['PKTIDX'],
            pktsize=header['PKTSIZE'], obsfreq=header['OBSFREQ'],
            src_name=header['SRC_NAME'], observer=header['OBSERVER'],
            telescop=header['TELESCOP'], ra_str=header['RA_STR'],
            dec_str=header['DEC_STR'])
        # There's a lot of extraneous stuff in the headers, so only test
        # values that were passed.
        assert header5.mutable is True
        assert header5.start_time == header.start_time
        # Tests PKTSIZE, etc. all at once.
        assert header5.offset == header.offset
        assert header5.sample_rate == header.sample_rate
        assert header5.overlap == header.overlap
        assert header5.samples_per_frame == header.samples_per_frame
        assert header5.sample_shape == header.sample_shape
        assert header5.sideband == header.sideband
        assert header5.bps == header.bps
        assert header5['OBSFREQ'] == header['OBSFREQ']
        assert header5['SRC_NAME'] == header['SRC_NAME']
        assert header5['OBSERVER'] == header['OBSERVER']
        assert header5['TELESCOP'] == header['TELESCOP']
        assert header5['RA_STR'] == header['RA_STR']
        assert header5['DEC_STR'] == header['DEC_STR']

        header_tuple = (((key, header[key]) for key in header))
        header6 = guppi.GUPPIHeader(header_tuple)
        assert header6 == header

        header7 = header.copy()
        assert header7 == header
        assert header7.mutable is True
        header8 = copy.copy(header)
        assert header8 == header
        assert header8.mutable is True

        # Check that we can set the offset by either passing the offset or the
        # start time.
        offset = 9.472 * u.s
        header9 = guppi.GUPPIHeader.fromvalues(
            time=header.start_time + offset, offset=offset,
            sample_rate=header.sample_rate,
            samples_per_frame=header.samples_per_frame,
            overlap=header.overlap, sample_shape=header.sample_shape,
            sideband=header.sideband, bps=header.bps,
            pktsize=header['PKTSIZE'], obsfreq=header['OBSFREQ'],
            src_name=header['SRC_NAME'], observer=header['OBSERVER'],
            telescop=header['TELESCOP'], ra_str=header['RA_STR'],
            dec_str=header['DEC_STR'])
        header10 = guppi.GUPPIHeader.fromvalues(
            start_time=header.start_time, time=header.start_time + offset,
            sample_rate=header.sample_rate,
            samples_per_frame=header.samples_per_frame,
            overlap=header.overlap, sample_shape=header.sample_shape,
            sideband=header.sideband, bps=header.bps,
            pktsize=header['PKTSIZE'], obsfreq=header['OBSFREQ'],
            src_name=header['SRC_NAME'], observer=header['OBSERVER'],
            telescop=header['TELESCOP'], ra_str=header['RA_STR'],
            dec_str=header['DEC_STR'])
        assert np.abs(header9.offset - offset) < 1 * u.ns
        assert np.abs(header9.start_time - header.start_time) < 1 * u.ns
        assert np.abs(header10.offset - offset) < 1 * u.ns
        assert np.abs(header10.start_time - header.start_time) < 1 * u.ns

    def test_payload(self, tmpdir):
        payload = self.payload
        assert payload.nbytes == 16384
        assert payload.shape == (1024, 2, 4)
        # Check sample shape validity.
        assert payload.sample_shape == (2, 4)
        assert payload.sample_shape.npol == 2
        assert payload.sample_shape.nchan == 4
        assert payload.size == 8192
        assert payload.ndim == 3
        assert payload.dtype == np.complex64
        assert np.all(payload[:3] == np.array(
            [[[-7.+12.j, -32.-10.j, -17.+25.j, 16.-5.j],
              [14.+21.j, -5.-7.j, 19.-8.j, 7.+7.j]],
             [[5.-3.j, -15.-14.j, -8.+14.j, -6.-18.j],
              [21.-1.j, 22.+6.j, -30.-13.j, 12.+23.j]],
             [[11.+2.j, 9.-13.j, 9.-15.j, -21.-6.j],
              [10.-12.j, -3.-10.j, -12.-8.j, 4.-27.j]]], dtype=np.complex64))
        assert np.all(payload[337:340] == np.array(
            [[[2.-25.j, 31.+2.j, -10.+1.j, -29.+14.j],
              [24.+6.j, -23.-16.j, -22.-20.j, -11.-6.j]],
             [[11.+10.j, -2.-1.j, -6.+9.j, 19.+16.j],
              [10.-25.j, -33.-5.j, 14.+0.j, 3.-3.j]],
             [[22.-7.j, 5.+11.j, -21.+4.j, 2.+0.j],
              [-4.-12.j, 1.+1.j, 13.+6.j, -31.-4.j]]], dtype=np.complex64))
        data = payload.data
        assert np.all(payload[:] == data)

        with open(str(tmpdir.join('testguppi.raw')), 'w+b') as s:
            payload.tofile(s)
            s.seek(0)
            payload2 = guppi.GUPPIPayload.fromfile(s, payload_nbytes=16384,
                                                   sample_shape=(2, 4), bps=8,
                                                   complex_data=True)
            assert payload2 == payload
            with pytest.raises(EOFError):
                # Too few bytes.
                s.seek(100)
                guppi.GUPPIPayload.fromfile(s, self.header)

        payload3 = guppi.GUPPIPayload.fromdata(payload.data, bps=8)
        assert payload3 == payload

        with open(SAMPLE_FILE, 'rb') as fh:
            fh.seek(self.header.nbytes)
            payload4 = guppi.GUPPIPayload.fromfile(fh, self.header,
                                                   memmap=True)
        assert isinstance(payload4.words, np.memmap)
        assert not isinstance(payload.words, np.memmap)
        assert payload == payload4

        # Check selective writing.
        payload5 = guppi.GUPPIPayload.fromdata(payload.data, bps=8)
        payload5[547:563, 0, :3] = (-1. + 3.j)
        assert np.all(payload5[547:563, 0, :3] == (-1. + 3.j))
        # Check nothing else has changed.
        assert np.all(payload5[:547] == payload[:547])
        assert np.all(payload5[563:] == payload[563:])
        assert np.all(payload5[547:563, 1] == payload[547:563, 1])
        assert np.all(payload5[547:563, 0, 3] == payload[547:563, 0, 3])
        some_data = np.array([5.-4.j, -2.+8.j], dtype=np.complex64)
        payload5[11:13, 1, 2] = some_data
        assert np.all(payload5[11:13, 1, 2] == some_data)
        # We can't yet read negative steps.
        with pytest.raises(AssertionError) as excinfo:
            payload5[27:13:-1, 1, 2]
        assert "cannot deal with negative steps" in str(excinfo.value)

        # Check that we can create and decode (nsample, nchan, npol) payloads.
        payload_tfirst = guppi.GUPPIPayload.fromdata(self.payload.data, bps=8,
                                                     channels_first=False)
        # First check channels_first makes a difference to payload words.
        assert ~np.all(payload_tfirst.words == payload.words)
        # Now check the data is the same.
        assert np.all(payload_tfirst.data == payload.data)
        # Check selective decode.
        item = (slice(547, 829, 2), slice(None), np.array([2, 1]))
        assert np.all(payload_tfirst[item] == payload[item])

    def test_file_reader(self):
        with guppi.open(SAMPLE_FILE, 'rb') as fh:
            header = fh.read_header()
            assert header == self.header
            current_pos = fh.tell()
            frame_rate = fh.get_frame_rate()
            assert frame_rate == (header.sample_rate /
                                  (header.samples_per_frame - header.overlap))
            assert fh.tell() == current_pos
            # Frame is done below, as is writing in binary.

    def test_frame(self, tmpdir):
        with guppi.open(SAMPLE_FILE, 'rb') as fh:
            frame = fh.read_frame(memmap=False)
        header, payload = frame.header, frame.payload
        assert header == self.header
        assert payload == self.payload
        assert frame == guppi.GUPPIFrame(header, payload)
        assert frame.sample_shape == payload.sample_shape
        assert frame.shape == (len(frame),) + frame.sample_shape
        assert frame.size == len(frame) * np.prod(frame.sample_shape)
        assert frame.ndim == payload.ndim
        assert np.all(frame[337:340] == np.array(
            [[[2.-25.j, 31.+2.j, -10.+1.j, -29.+14.j],
              [24.+6.j, -23.-16.j, -22.-20.j, -11.-6.j]],
             [[11.+10.j, -2.-1.j, -6.+9.j, 19.+16.j],
              [10.-25.j, -33.-5.j, 14.+0.j, 3.-3.j]],
             [[22.-7.j, 5.+11.j, -21.+4.j, 2.+0.j],
              [-4.-12.j, 1.+1.j, 13.+6.j, -31.-4.j]]], dtype=np.complex64))

        with open(str(tmpdir.join('testguppi.raw')), 'w+b') as s:
            frame.tofile(s)
            s.seek(0)
            frame2 = guppi.GUPPIFrame.fromfile(s, memmap=False)
        assert frame2 == frame
        frame3 = guppi.GUPPIFrame.fromdata(payload.data, header)
        assert frame3 == frame
        frame4 = guppi.GUPPIFrame.fromdata(payload.data, **header)
        assert frame4 == frame
        header5 = header.copy()
        frame5 = guppi.GUPPIFrame(header5, payload, valid=False)
        assert frame5.valid is False
        assert np.all(frame5.data == 0.)
        invalid_samples = frame5[-1000:]
        assert np.all(invalid_samples == 0.)
        assert invalid_samples.shape == (1000, 2, 4)
        invalid_samples = frame5[8192:]
        assert invalid_samples.shape == (0, 2, 4)
        frame5.valid = True
        assert frame5 == frame

    def test_frame_memmap(self, tmpdir):
        # Get frame the regular way.
        with guppi.open(SAMPLE_FILE, 'rb') as fr:
            frame = fr.read_frame(memmap=False)
        assert not isinstance(frame.payload.words, np.memmap)
        # Check that if we map it instead, we get the same result.
        with guppi.open(SAMPLE_FILE, 'rb') as fh:
            frame2 = fh.read_frame(memmap=True)
        assert frame2 == frame
        assert isinstance(frame2.payload.words, np.memmap)
        # Bit superfluous perhaps, but check decoding as well.
        assert np.all(frame2[337:340] == np.array(
            [[[2.-25.j, 31.+2.j, -10.+1.j, -29.+14.j],
              [24.+6.j, -23.-16.j, -22.-20.j, -11.-6.j]],
             [[11.+10.j, -2.-1.j, -6.+9.j, 19.+16.j],
              [10.-25.j, -33.-5.j, 14.+0.j, 3.-3.j]],
             [[22.-7.j, 5.+11.j, -21.+4.j, 2.+0.j],
              [-4.-12.j, 1.+1.j, 13.+6.j, -31.-4.j]]], dtype=np.complex64))
        assert np.all(frame2.data == frame.data)

        # Now check writing.  First, without memmap, just ensuring writing
        # to file works as well as to BytesIO done above.
        filename = str(tmpdir.join('testguppi.raw'))
        with guppi.open(filename, 'wb') as fw:
            fw.write_frame(frame)

        with guppi.open(filename, 'rb') as fw:
            frame3 = fw.read_frame()

        assert frame3 == frame
        # Now memmap file to be written to.
        with guppi.open(filename, 'wb') as fw:
            frame4 = fw.memmap_frame(frame.header)
        # Initially no data set, so frames should not match yet.
        assert frame4 != frame
        # So, if we read this file, it also should not match.
        with guppi.open(filename, 'rb') as fw:
            frame5 = fw.read_frame()
        assert frame5 != frame

        # Fill in some data.  This should only update some words.
        frame4[:20] = frame[:20]
        assert np.all(frame4[:20] == frame[:20])
        assert frame4 != frame
        # Update the rest, so it becomes the same.
        frame4[20:] = frame[20:]
        assert frame4 == frame
        # Delete to flush to disk just to be sure, then read and check it.
        del frame4
        with guppi.open(filename, 'rb') as fn:
            frame6 = fn.read_frame()

        assert frame6 == frame
        # Some further tests for completeness;
        # initiate frame using data and header keywords.
        with guppi.open(filename, 'wb') as fw:
            fw.write_frame(self.payload.data, **self.header)
        with guppi.open(filename, 'rb') as fh:
            frame7 = fh.read_frame()
        assert frame7 == frame
        # memmap frame using header keywords.
        with guppi.open(filename, 'wb') as fw:
            frame8 = fw.memmap_frame(**self.header)
            frame8[:] = self.payload.data
        assert frame8 == frame
        del frame8
        with guppi.open(filename, 'rb') as fh:
            frame9 = fh.read_frame()
        assert frame9 == frame

    def test_filestreamer(self, tmpdir):
        start_time = self.header.time
        with guppi.open(SAMPLE_FILE, 'rs') as fh:
            assert fh.header0 == self.header
            assert fh.sample_shape == (2, 4)
            assert fh.shape == (3840,) + fh.sample_shape
            assert fh.size == np.prod(fh.shape)
            assert fh.ndim == len(fh.shape)
            assert fh.start_time == start_time
            assert fh.sample_rate == 250 * u.Hz
            record = fh.read()
            fh.seek(0)
            record1 = fh.read(12)
            assert fh.tell() == 12
            fh.seek(1523)
            record2 = np.zeros((2, 2, 4), dtype=np.complex64)
            record2 = fh.read(out=record2)
            # Check that stream properly skips overlap.
            assert np.all(record2 == fh._frame[563:565])
            assert fh.tell() == 1525
            assert fh.time == fh.tell(unit='time')
            assert (np.abs(fh.time - (start_time + 1525 / (250 * u.Hz))) <
                    1. * u.ns)
            fh.seek(fh.start_time + 100 / (250 * u.Hz))
            assert fh.tell() == 100
            assert (np.abs(fh.stop_time -
                           (start_time + 3840 / (250 * u.Hz))) < 1. * u.ns)
            fh.seek(1, 'end')
            with pytest.raises(EOFError):
                fh.read()

        assert record1.shape == (12, 2, 4)
        assert np.all(record1[:3] == np.array(
            [[[-7.+12.j, -32.-10.j, -17.+25.j, 16.-5.j],
              [14.+21.j, -5.-7.j, 19.-8.j, 7.+7.j]],
             [[5.-3.j, -15.-14.j, -8.+14.j, -6.-18.j],
              [21.-1.j, 22.+6.j, -30.-13.j, 12.+23.j]],
             [[11.+2.j, 9.-13.j, 9.-15.j, -21.-6.j],
              [10.-12.j, -3.-10.j, -12.-8.j, 4.-27.j]]], dtype=np.complex64))
        assert record1.dtype == np.complex64
        assert np.all(record1 == self.payload[:12].squeeze())
        assert record2.shape == (2, 2, 4)

        filename = str(tmpdir.join('testguppi.raw'))
        spf = self.header.samples_per_frame - self.header.overlap
        # Need to remove overlap.
        with guppi.open(filename, 'ws', header0=self.header_w,
                        squeeze=False) as fw:
            assert fw.sample_rate == 250 * u.Hz
            fw.write(self.payload[:spf])
            assert fw.start_time == start_time
            assert (np.abs(fw.time - (start_time + spf / (250 * u.Hz))) <
                    1. * u.ns)

        with guppi.open(filename, 'rs') as fh:
            data = fh.read()
            assert fh.start_time == start_time
            assert (np.abs(fh.time - (start_time + spf / (250 * u.Hz))) <
                    1. * u.ns)
            assert fh.stop_time == fh.time
            assert fh.sample_rate == 250 * u.Hz
        assert np.all(data == self.payload[:spf].squeeze())

        # Try single polarisation (two channels to keep it complex), and check
        # initialisation by header keywords.  This also tests writing
        # with squeeze=True.
        h = self.header
        with guppi.open(filename, 'ws', time=h.time, bps=h.bps,
                        sample_rate=h.sample_rate,
                        pktsize=h['PKTSIZE'], overlap=0,
                        payload_nbytes=self.header_w.payload_nbytes // 4,
                        nchan=2, npol=1) as fw:
            fw.write(self.payload[:spf, 0, :2])
            assert np.abs(fw.start_time - start_time) < 1.*u.ns
            assert (np.abs(fw.time - (start_time + spf / (250 * u.Hz))) <
                    1. * u.ns)

        with guppi.open(filename, 'rs') as fh:
            data_onepol = fh.read()
            assert np.abs(fh.start_time - start_time) < 1.*u.ns
            assert np.abs(fh.stop_time - (start_time + spf /
                                          (250 * u.Hz))) < 1.*u.ns
        assert np.all(data_onepol == self.payload[:spf, 0, :2])

        # Try reading a single polarization.
        with guppi.open(SAMPLE_FILE, 'rs', subset=0) as fh:
            assert fh.sample_shape == (4,)
            assert fh.subset == (0,)
            record3 = fh.read(12)
            assert np.all(record3 == record[:12, 0, :4])

        # Read right polarization, but read channels in reverse order.
        with guppi.open(SAMPLE_FILE, 'rs', subset=(1, [1, 0])) as fh:
            assert fh.sample_shape == (2,)
            data_sub = fh.read()
            assert np.all(data_sub[:, 1] == record[:, 1, 0])

        # Test that squeeze attribute works on read (including in-place read).
        with guppi.open(filename, 'rs', squeeze=False) as fh:
            assert fh.sample_shape == (1, 2)
            assert fh.sample_shape.npol == 1
            assert fh.sample_shape.nchan == 2
            assert fh.read(1).shape == (1, 1, 2)
            assert fh.read(10).shape == (10, 1, 2)
            fh.seek(0)
            out = np.zeros((12, 1, 2), dtype=np.complex64)
            fh.read(out=out)
            assert fh.tell() == 12
            assert np.all(out.squeeze() == self.payload[:12, 0, :2])

        # Test that squeeze=False works on write (we checked above it works
        # with squeeze=True).
        with guppi.open(filename, 'ws', time=h.time, bps=h.bps,
                        sample_rate=h.sample_rate,
                        pktsize=h['PKTSIZE'], overlap=0,
                        payload_nbytes=self.header_w.payload_nbytes // 4,
                        nchan=2, npol=1, squeeze=False) as fw:
            fw.write(self.payload[:spf, 0:1, :2])
            assert np.abs(fw.start_time - start_time) < 1.*u.ns
            assert (np.abs(fw.time - (start_time + spf / (250 * u.Hz))) <
                    1. * u.ns)

        with guppi.open(filename, 'rs', squeeze=False) as fh:
            data_onepol = fh.read()
        assert np.all(data_onepol.squeeze() == self.payload[:spf, 0, :2])

    def test_incomplete_stream(self, tmpdir):
        """Test that writing an incomplete stream is possible, and that frame
        set is valid but invalid samples use the fill value.
        """
        filename = str(tmpdir.join('testguppi.raw'))
        with catch_warnings(UserWarning) as w:
            with guppi.open(filename, 'ws', header0=self.header_w,
                            squeeze=False) as fw:
                fw.write(self.payload[:10])
        assert len(w) == 1
        assert 'partial buffer' in str(w[0].message)
        with guppi.open(filename, 'rs', squeeze=False) as fwr:
            data = fwr.read()
            assert np.all(data[:10] == self.payload[:10])
            assert np.all(data[10:] == fwr.fill_value)

    def test_multiple_files_stream(self, tmpdir):
        start_time = self.header_w.time
        with guppi.open(SAMPLE_FILE, 'rs') as fh:
            data = fh.read()
        filenames = (str(tmpdir.join('guppi_1.raw')),
                     str(tmpdir.join('guppi_2.raw')))
        with guppi.open(filenames, 'ws', header0=self.header_w,
                        frames_per_file=2) as fw:
            start_time = fw.start_time
            fw.write(data[:1000])
            time1000 = fw.time
            fw.write(data[1000:])
            stop_time = fw.time
        assert start_time == self.header_w.time
        assert np.abs(time1000 -
                      (start_time + 1000 / (250 * u.Hz))) < 1.*u.ns
        assert np.abs(stop_time -
                      (start_time + 3840 / (250 * u.Hz))) < 1.*u.ns

        with guppi.open(filenames[1], 'rs') as fr:
            assert (np.abs(fr.time - (start_time + 1920 / (250 * u.Hz))) <
                    1. * u.ns)
            data1 = fr.read()
        assert np.all(data1 == data[1920:])

        with guppi.open(filenames, 'rs') as fr:
            assert fr.start_time == start_time
            assert fr.time == start_time
            assert np.abs(fr.stop_time -
                          (start_time + 3840 / (250 * u.Hz))) < 1. * u.ns
            data2 = fr.read()
            assert np.abs(fr.time - fr.stop_time) < 1. * u.ns
        assert np.all(data2 == data)

        # Pass sequentialfile objects to reader.
        with sf.open(filenames, 'w+b',
                     file_size=2*self.header_w.frame_nbytes) as fraw, \
                guppi.open(fraw, 'ws', header0=self.header_w) as fw:
            fw.write(data)

        with sf.open(filenames, 'rb') as fraw, \
                guppi.open(fraw, 'rs') as fr:
            data3 = fr.read()
        assert np.all(data3 == data)

        # Check that we can't pass a filename sequence in 'wb' mode.
        with pytest.raises(ValueError):
            guppi.open(filenames, 'wb')

    def test_partial_last_frame(self, tmpdir):
        """Test reading a file with an incomplete last frame."""
        # Read in sample file as a byte stream.
        with guppi.open(SAMPLE_FILE, 'rb') as fh:
            puppi_raw = fh.read()

        # Try reading a file with an incomplete payload.
        with guppi.open(str(tmpdir.join('puppi_partframe.raw')), 'wb') as fw:
            fw.write(puppi_raw[:len(puppi_raw) - 6091])
        with guppi.open(str(tmpdir.join('puppi_partframe.raw')), 'rs') as fn:
            assert fn.shape == (2880, 2, 4)
            assert np.abs(fn.stop_time - fn.start_time -
                          2880 / (250 * u.Hz)) < 1. * u.ns

        # Try reading a file with an incomplete header.
        with guppi.open(str(tmpdir.join('puppi_partframe.raw')), 'wb') as fw:
            fw.write(puppi_raw[:len(puppi_raw) - 17605])
        with guppi.open(str(tmpdir.join('puppi_partframe.raw')), 'rs') as fn:
            assert fn.shape == (2880, 2, 4)
            assert np.abs(fn.stop_time - fn.start_time -
                          2880 / (250 * u.Hz)) < 1. * u.ns

    def test_chan_ordered_stream(self, tmpdir):
        """Test encoding and decoding frames and streams that use
        (nsample, nchan, npol).
        """
        filename = str(tmpdir.join('testguppi.raw'))

        with guppi.open(SAMPLE_FILE) as fh:
            data = fh.read()

        header = self.header.copy()
        header.channels_first = False
        header['OVERLAP'] = 0
        header.samples_per_frame = 960
        with guppi.open(filename, 'ws', header0=header) as fw:
            fw.write(data)
        with guppi.open(filename) as fn:
            fn.seek(1231)
            new_data = fn.read(47)
            assert np.all(new_data == data[1231:1231 + 47])

    def test_template_stream(self, tmpdir):
        start_time = self.header_w.time
        with guppi.open(SAMPLE_FILE, 'rs') as fh:
            data = fh.read()

        # Simple template with file number counter.
        template = str(tmpdir.join('guppi_{file_nr:02d}.raw'))
        with guppi.open(template, 'ws', frames_per_file=1,
                        **self.header_w) as fw:
            fw.write(data)

        with guppi.open(template, 'rs') as fr:
            assert len(fr.fh_raw.files) == 4
            assert fr.fh_raw.files[-1] == str(tmpdir.join('guppi_03.raw'))
            assert np.abs(fr.stop_time -
                          (start_time + 3840 / (250 * u.Hz))) < 1.*u.ns
            data2 = fr.read()
        assert np.all(data2 == data)

        # More complex template that requires keywords.
        template = str(tmpdir.join('puppi_{stt_imjd}.{file_nr:04d}.raw'))
        with guppi.open(template, 'ws', frames_per_file=1,
                        header0=self.header_w) as fw:
            fw.write(data[:1920])
            assert fw.start_time == start_time
            assert (np.abs(fw.time - (start_time + 1920 / (250 * u.Hz))) <
                    1. * u.ns)
            fw.write(data[1920:])
            assert (np.abs(fw.time - (start_time + 3840 / (250 * u.Hz))) <
                    1. * u.ns)

        # We cannot just open using the same template, since STT_IMJD is
        # not available.
        with pytest.raises(KeyError):
            guppi.open(template, 'rs')

        kwargs = dict(STT_IMJD=self.header_w['STT_IMJD'])
        with guppi.open(template, 'rs', **kwargs) as fr:
            data3 = fr.read()
        assert np.all(data3 == data)

        # Try passing stream reader kwargs.
        with guppi.open(template, 'rs', subset=(0, [2, 3]), squeeze=False,
                        **kwargs) as fr:
            data4 = fr.read()
        assert np.all(data4.squeeze() == data[:, 0, 2:])

        # Just to check internal checks are OK.
        filename = template.format(stt_imjd=self.header_w['STT_IMJD'],
                                   file_nr=0)
        with pytest.raises(ValueError):
            guppi.open(filename, 's')
        with pytest.raises(TypeError):
            # Extraneous argument.
            guppi.open(filename, 'rs', files=(filename,))


class TestGUPPIFileNameSequencer(object):
    def setup(self):
        with open(SAMPLE_FILE, 'rb') as fh:
            self.header = guppi.GUPPIHeader.fromfile(fh)

    def test_header_extraction(self):
        # Follow Nikhil's J1810 Arecibo observation file naming scheme:
        # puppi_58132_J1810+1744_2176.0000.raw, etc.
        template = 'puppi_{stt_imjd}_{src_name}_{scannum}.{file_nr:04d}.raw'
        fns = GUPPIFileNameSequencer(template, self.header)
        assert fns[0] == 'puppi_58132_J1810+1744_2176.0000.raw'
        assert fns[29] == 'puppi_58132_J1810+1744_2176.0029.raw'
