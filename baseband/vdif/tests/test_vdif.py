# Licensed under the GPLv3 - see LICENSE
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import pytest
from astropy.time import Time
import astropy.units as u
from astropy.tests.helper import catch_warnings

from ... import vdif, vlbi_base
from ...helpers import sequentialfile as sf
from ...data import (SAMPLE_VDIF as SAMPLE_FILE, SAMPLE_VLBI_VDIF as
                     SAMPLE_VLBI, SAMPLE_MWA_VDIF as SAMPLE_MWA,
                     SAMPLE_AROCHIME_VDIF as SAMPLE_AROCHIME)

# Comparison with m5access routines (check code on 2015-MAY-30) on vlba.m5a,
# which contains the first 16 frames from evn/Fd/GP052D_FD_No0006.m5a.
# 00000000  77 2c db 00 00 00 00 1c  75 02 00 20 fc ff 01 04  # header 0 - 3
# 00000010  10 00 80 03 ed fe ab ac  00 00 40 33 83 15 03 f2  # header 4 - 7
# 00000020  2a 0a 7c 43 8b 69 9d 59  cb 99 6d 9a 99 96 5d 67  # data 0 - 3
# NOTE: thread_id = 1
# 2a = 00 10 10 10 = (lsb first) 1,  1,  1, -3
# 0a = 00 00 10 10 =             1,  1, -3, -3
# 7c = 01 11 11 00 =            -3,  3,  3, -1
# m5d evn/Fd/GP052D_FD_No0006.m5a VDIF_5000-512-1-2 100
# Mark5 stream: 0x16cd140
#   stream = File-1/1=evn/Fd/GP052D_FD_No0006.m5a
#   format = VDIF_5000-512-1-2 = 3
#   start mjd/sec = 56824 21367.000000000
#   frame duration = 78125.00 ns
#   framenum = 0
#   sample rate = 256000000 Hz
#   offset = 0
#   framebytes = 5032 bytes
#   datasize = 5000 bytes
#   sample granularity = 4
#   frame granularity = 1
#   gframens = 78125
#   payload offset = 32
#   read position = 0
#   data window size = 1048576 bytes


class TestVDIF(object):

    def test_header(self, tmpdir):
        with open(SAMPLE_FILE, 'rb') as fh:
            header = vdif.VDIFHeader.fromfile(fh)
        assert header.nbytes == 32
        assert header.edv == 3
        mjd, frac = divmod(header.time.mjd, 1)
        assert mjd == 56824
        assert round(frac * 86400) == 21367
        assert header.payload_nbytes == 5000
        assert header.frame_nbytes == 5032
        assert header['thread_id'] == 1
        assert header.sample_rate == 32*u.MHz
        assert header.samples_per_frame == 20000
        assert header.nchan == 1
        assert header.bps == 2
        assert not header['complex_data']
        assert header.mutable is False
        with open(str(tmpdir.join('test.vdif')), 'w+b') as s:
            header.tofile(s)
            s.seek(0)
            header2 = vdif.VDIFHeader.fromfile(s)
        assert header2 == header
        assert header2.mutable is False
        header3 = vdif.VDIFHeader.fromkeys(**header)
        assert header3 == header
        assert header3.mutable is True
        with pytest.raises(KeyError):
            vdif.VDIFHeader.fromkeys(extra=1, **header)
        with pytest.raises(KeyError):
            kwargs = dict(header)
            kwargs.pop('thread_id')
            vdif.VDIFHeader.fromkeys(**kwargs)
        # Try initialising with properties instead of keywords, as much as
        # possible.  Note that we still have to give a lots of extra, less
        # directly useful parameters to get an *identical* header.
        header4 = vdif.VDIFHeader.fromvalues(
            edv=header.edv, ref_epoch=header['ref_epoch'],
            seconds=header['seconds'], frame_nr=header['frame_nr'],
            samples_per_frame=header.samples_per_frame,
            bps=header.bps, complex_data=header['complex_data'],
            thread_id=header['thread_id'], station=header.station,
            sampling_unit=header['sampling_unit'],
            sampling_rate=header['sampling_rate'],
            loif_tuning=header['loif_tuning'], dbe_unit=header['dbe_unit'],
            if_nr=header['if_nr'], subband=header['subband'],
            sideband=header['sideband'], major_rev=header['major_rev'],
            minor_rev=header['minor_rev'], personality=header['personality'],
            _7_28_4=header['_7_28_4'])
        # The same header, but created by passing time and sample rate.
        header4_usetime = vdif.VDIFHeader.fromvalues(
            edv=header.edv, time=header.time,
            samples_per_frame=header.samples_per_frame,
            station=header.station, sample_rate=header.sample_rate,
            bps=header.bps, complex_data=header['complex_data'],
            thread_id=header['thread_id'],
            loif_tuning=header['loif_tuning'], dbe_unit=header['dbe_unit'],
            if_nr=header['if_nr'], subband=header['subband'],
            sideband=header['sideband'], major_rev=header['major_rev'],
            minor_rev=header['minor_rev'], personality=header['personality'],
            _7_28_4=header['_7_28_4'])
        assert header4 == header
        assert header4.mutable is True
        assert header4 == header4_usetime
        header5 = header.copy()
        assert header5 == header
        assert header5.mutable is True
        header5['thread_id'] = header['thread_id'] + 1
        assert header5['thread_id'] == header['thread_id'] + 1
        assert header5 != header
        with pytest.raises(TypeError):
            header['thread_id'] = 0
        # Also test time setting.
        header5.time = header.time + 1.*u.s
        frame_rate = header.sample_rate / header.samples_per_frame
        assert abs(header5.time - header.time - 1.*u.s) < 1.*u.ns
        assert header5['frame_nr'] == header['frame_nr']
        header5.time = header.time + 1.*u.s + 1.1/frame_rate
        assert abs(header5.time - header.time - 1.*u.s -
                   1./frame_rate) < 1.*u.ns
        assert header5['frame_nr'] == header['frame_nr'] + 1
        # Check rounding in corner case.
        header5.time = header.time + 1.*u.s - 0.01/frame_rate
        assert abs(header5.time - header.time - 1.*u.s) < 1.*u.ns
        assert header5['frame_nr'] == header['frame_nr']
        # Check requesting non-existent EDV returns VDIFBaseHeader instance
        header6 = vdif.header.VDIFHeader.fromvalues(edv=100)
        assert type(header6) is vdif.header.VDIFBaseHeader
        assert header6['edv'] == 100

        # Make a new header to test passing time/sample rate.
        headerT = header.copy()
        headerT.time = header.time + 1. / frame_rate

        # Test initializing EDV 0 with properties, but off of 1 second mark so
        # frame_nr is used.
        header8 = vdif.VDIFHeader.fromvalues(
            edv=0, ref_epoch=headerT['ref_epoch'], seconds=headerT['seconds'],
            frame_nr=headerT['frame_nr'], complex_data=headerT['complex_data'],
            samples_per_frame=headerT.samples_per_frame, bps=headerT.bps,
            station=headerT.station, thread_id=headerT['thread_id'])
        assert header8['ref_epoch'] == headerT['ref_epoch']
        assert header8['seconds'] == headerT['seconds']
        assert header8['frame_nr'] == headerT['frame_nr']
        # The same header, but created by passing time and sample rate.
        header8_usetime = vdif.VDIFHeader.fromvalues(
            edv=0, time=headerT.time, sample_rate=headerT.sample_rate,
            complex_data=headerT['complex_data'], bps=headerT.bps,
            samples_per_frame=headerT.samples_per_frame,
            station=headerT.station, thread_id=headerT['thread_id'])
        assert header8_usetime == header8

        # Without a sample rate or frame_nr, cannot initialize using time.
        with pytest.raises(ValueError):
            vdif.VDIFHeader.fromvalues(
                edv=0, time=headerT.time, complex_data=headerT['complex_data'],
                bps=headerT.bps, samples_per_frame=headerT.samples_per_frame,
                station=headerT.station, thread_id=headerT['thread_id'])

        # Without a sample rate for EDV 1, 3, cannot initialize using time.
        with pytest.raises(ValueError):
            vdif.VDIFHeader.fromvalues(
                edv=1, time=headerT.time, station=headerT.station,
                samples_per_frame=headerT.samples_per_frame,
                bps=headerT.bps, complex_data=headerT['complex_data'],
                thread_id=headerT['thread_id'])

        # Check rounding in corner case.
        header9 = headerT.copy()
        header9.set_time(Time('2018-01-01T00:34:07.999999999996'))
        assert header9['seconds'] == 2048
        assert header9['frame_nr'] == 0

        # Check that with a missing sampling rate, we do not return a wrong
        # time.
        header10 = headerT.copy()
        header10['sampling_rate'] = 0
        with pytest.raises(ValueError):
            header10.time
        assert (header10.get_time(frame_rate=headerT.frame_rate) ==
                headerT.time)

    def test_custom_header(self, tmpdir):
        # Custom header with an EDV that already exists
        with pytest.raises(ValueError):
            class VDIFHeaderY(vdif.header.VDIFBaseHeader):
                _edv = 3
        # Custom header that has neglected to override _edv
        with pytest.raises(ValueError):
            class VDIFHeaderZ(vdif.header.VDIFBaseHeader):
                pass

        # Working header with nonsense data in the last two words.
        class VDIFHeaderX(vdif.header.VDIFSampleRateHeader):
            _edv = 0x58
            _header_parser = (vdif.header.VDIFSampleRateHeader._header_parser +
                              vlbi_base.header.HeaderParser(
                                  (('nonsense_0', (6, 0, 32, 0x0)),
                                   ('nonsense_1', (7, 0, 8, None)),
                                   ('nonsense_2', (7, 8, 24, 0x1)))))

            def verify(self):
                super(VDIFHeaderX, self).verify()

        assert vdif.header.VDIF_HEADER_CLASSES[0x58] is VDIFHeaderX

        # Read in a header, and hack an 0x58 header with its info
        with open(SAMPLE_FILE, 'rb') as fh:
            header = vdif.VDIFHeader.fromfile(fh)

        headerX_dummy = vdif.VDIFHeader.fromvalues(
            edv=0x58, time=header.time,
            samples_per_frame=header.samples_per_frame,
            station=header.station, sample_rate=header.sample_rate,
            bps=header.bps, complex_data=header['complex_data'],
            thread_id=header['thread_id'], nonsense_0=2000000000,
            nonsense_1=100, nonsense_2=10000000)

        # Check that headerX_dummy is indeed VDIFHeaderX class, with
        # the correct EDV
        assert isinstance(headerX_dummy, VDIFHeaderX)
        assert headerX_dummy.edv == 0x58

        # Write to dummy file, then re-read.
        with open(str(tmpdir.join('test.vdif')), 'w+b') as s:
            headerX_dummy.tofile(s)
            s.seek(0)
            headerX = vdif.VDIFHeader.fromfile(s)

        # Ensure values have been copied and re-read properly - only
        # edv, mutability and nonsense values should have changed or
        # been added.
        assert isinstance(headerX, VDIFHeaderX)
        assert headerX.nbytes == header.nbytes
        assert headerX.edv == 0x58
        mjd, frac = divmod(header.time.mjd, 1)
        assert mjd == 56824
        assert round(frac * 86400) == 21367
        assert headerX.payload_nbytes == header.payload_nbytes
        assert headerX.frame_nbytes == header.frame_nbytes
        assert headerX['thread_id'] == header['thread_id']
        assert headerX.sample_rate == header.sample_rate
        assert headerX.samples_per_frame == header.samples_per_frame
        assert headerX.nchan == header.nchan
        assert headerX.bps == header.bps
        assert not headerX['complex_data']
        assert headerX.mutable is False
        assert headerX['nonsense_0'] == 2000000000
        assert headerX['nonsense_1'] == 100
        assert headerX['nonsense_2'] == 10000000

    def test_decoding(self, tmpdir):
        """Check that look-up levels are consistent with mark5access."""
        o2h = vlbi_base.encoding.OPTIMAL_2BIT_HIGH
        assert np.all(vdif.payload.lut1bit[0] == -1.)
        assert np.all(vdif.payload.lut1bit[0xff] == 1.)
        assert np.all(vdif.payload.lut1bit.astype(int) ==
                      ((np.arange(256)[:, np.newaxis] >>
                        np.arange(8)) & 1) * 2 - 1)
        assert np.all(vdif.payload.lut2bit[0] == -o2h)
        assert np.all(vdif.payload.lut2bit[0x55] == -1.)
        assert np.all(vdif.payload.lut2bit[0xaa] == 1.)
        assert np.all(vdif.payload.lut2bit[0xff] == o2h)
        assert np.allclose(vdif.payload.lut4bit[0] * 2.95, -8.)
        assert np.all(vdif.payload.lut4bit[0x88] == 0.)
        assert np.allclose(vdif.payload.lut4bit[0xff] * 2.95, 7)
        assert np.all(-vdif.payload.lut4bit[0x11] ==
                      vdif.payload.lut4bit[0xff])
        aint = np.arange(0, 256, dtype=np.uint8)
        words = aint.view('<u4')
        areal = np.linspace(-127.5, 127.5, 256).reshape(-1, 1) / 35.5
        acmplx = areal[::2] + 1j * areal[1::2]
        payload1 = vdif.VDIFPayload(words, bps=8, complex_data=False)
        assert np.allclose(payload1.data, areal)
        payload2 = vdif.VDIFPayload(words, bps=8, complex_data=True)
        assert np.allclose(payload2.data, acmplx)
        header = vdif.VDIFHeader.fromvalues(edv=0, complex_data=False, bps=8,
                                            payload_nbytes=payload1.nbytes)
        assert vdif.VDIFPayload.fromdata(areal, header) == payload1
        # Also check that a circular decode-encode is self-consistent.
        assert vdif.VDIFPayload.fromdata(payload1.data, header) == payload1
        header['complex_data'] = True
        assert vdif.VDIFPayload.fromdata(acmplx, header) == payload2
        assert vdif.VDIFPayload.fromdata(payload2.data, header) == payload2
        # Also check for bps=4.
        decode = (aint[:, np.newaxis] >> np.array([0, 4])) & 0xf
        areal = ((decode - 8.) / 2.95).reshape(-1, 1)
        acmplx = areal[::2] + 1j * areal[1::2]
        payload3 = vdif.VDIFPayload(words, bps=4, complex_data=False)
        assert np.allclose(payload3.data, areal)
        payload4 = vdif.VDIFPayload(words, bps=4, complex_data=True)
        assert np.allclose(payload4.data, acmplx)
        header = vdif.VDIFHeader.fromvalues(edv=0, complex_data=False, bps=4,
                                            payload_nbytes=payload3.nbytes)
        assert vdif.VDIFPayload.fromdata(areal, header) == payload3
        assert vdif.VDIFPayload.fromdata(payload3.data, header) == payload3
        header['complex_data'] = True
        assert vdif.VDIFPayload.fromdata(acmplx, header) == payload4
        assert vdif.VDIFPayload.fromdata(payload4.data, header) == payload4

    def test_payload(self, tmpdir):
        with open(SAMPLE_FILE, 'rb') as fh:
            header = vdif.VDIFHeader.fromfile(fh)
            payload = vdif.VDIFPayload.fromfile(fh, header)
        assert payload.nbytes == 5000
        assert payload.shape == (20000, 1)
        assert payload.size == 20000
        assert payload.ndim == 2
        # Check sample shape validity
        assert payload.sample_shape == (1,)
        assert payload.sample_shape.nchan == 1
        assert payload.dtype == np.float32
        assert np.all(payload[:12, 0].astype(int) ==
                      np.array([1, 1, 1, -3, 1, 1, -3, -3, -3, 3, 3, -1]))
        with open(str(tmpdir.join('test.vdif')), 'w+b') as s:
            payload.tofile(s)
            s.seek(0)
            payload2 = vdif.VDIFPayload.fromfile(s, header)
            assert payload2 == payload
            with pytest.raises(EOFError):
                # Too few bytes.
                s.seek(100)
                vdif.VDIFPayload.fromfile(s, header)
        payload3 = vdif.VDIFPayload.fromdata(payload.data, header)
        assert payload3 == payload
        with pytest.raises(ValueError):
            # Wrong number of channels.
            vdif.VDIFPayload.fromdata(np.empty((payload.shape[0], 2)), header)
        with pytest.raises(ValueError):
            # Too few data.
            vdif.VDIFPayload.fromdata(payload[:100], header)
        # check if it works with complex data
        payload4 = vdif.VDIFPayload(payload.words, nchan=1, bps=2,
                                    complex_data=True)
        assert payload4.complex_data is True
        assert payload4.nbytes == 5000
        assert payload4.shape == (10000, 1)
        assert payload4.dtype == np.complex64
        assert np.all(payload4.data ==
                      payload[::2] + 1j * payload[1::2])
        with pytest.raises(ValueError):
            vdif.VDIFPayload.fromdata(payload4.data, header)
        header5 = header.copy()
        header5['complex_data'] = True
        payload5 = vdif.VDIFPayload.fromdata(payload4.data, header5)
        assert payload5 == payload4

    @pytest.mark.parametrize('item', (2, (), -1, slice(1, 3),
                                      slice(2, 4), slice(-3, None)))
    def test_payload_getitem_setitem(self, item):
        with open(SAMPLE_FILE, 'rb') as fh:
            header = vdif.VDIFHeader.fromfile(fh)
            payload = vdif.VDIFPayload.fromfile(fh, header)
        sel_data = payload.data[item]
        assert np.all(payload[item] == sel_data)
        payload2 = vdif.VDIFPayload(payload.words.copy(), header)
        assert payload2 == payload
        payload2[item] = -sel_data
        check = payload.data
        check[item] = -sel_data
        assert np.all(payload2[item] == -sel_data)
        assert np.all(payload2.data == check)
        assert payload2 != payload
        payload2[item] = sel_data
        assert np.all(payload2[item] == sel_data)
        assert payload2 == payload

    def test_filereader(self):
        with vdif.open(SAMPLE_FILE, 'rb') as fh:
            header = vdif.VDIFHeader.fromfile(fh)
            fh.seek(0)
            header2 = fh.read_header()
            assert header2 == header
            current_pos = fh.tell()
            frame_rate = fh.get_frame_rate()
            assert abs(frame_rate - 32. * u.MHz /
                       header.samples_per_frame) < 1. * u.nHz
            assert fh.tell() == current_pos
            # The read_frame method is tested below, as is mode='wb'.

    def test_frame(self, tmpdir):
        with vdif.open(SAMPLE_FILE, 'rb') as fh:
            header = vdif.VDIFHeader.fromfile(fh)
            payload = vdif.VDIFPayload.fromfile(fh, header)
            fh.seek(0)
            frame = fh.read_frame()

        assert frame.header == header
        assert frame.payload == payload
        assert frame.shape == payload.shape
        assert frame.size == payload.size
        assert frame.ndim == payload.ndim
        assert frame == vdif.VDIFFrame(header, payload)
        assert np.all(frame.data[:12, 0].astype(int) ==
                      np.array([1, 1, 1, -3, 1, 1, -3, -3, -3, 3, 3, -1]))
        vdif_test = str(tmpdir.join('test.vdif'))
        with open(vdif_test, 'w+b') as s:
            frame.tofile(s)
            s.seek(0)
            frame2 = vdif.VDIFFrame.fromfile(s)

        assert frame2 == frame
        frame3 = vdif.VDIFFrame.fromdata(payload.data, header)
        assert frame3 == frame
        frame4 = vdif.VDIFFrame.fromdata(payload.data, **header)
        assert frame4 == frame
        header5 = header.copy()
        frame5 = vdif.VDIFFrame(header5, payload, valid=False)
        assert frame5.valid is False
        assert np.all(frame5.data == 0.)
        frame5.valid = True
        assert frame5 == frame
        # Also test binary file writer.
        with vdif.open(vdif_test, 'wb') as fw:
            fw.write_frame(frame)
            fw.write_frame(frame.data, header)
        with open(vdif_test, 'rb') as s:
            frame6 = vdif.VDIFFrame.fromfile(s)
            frame7 = vdif.VDIFFrame.fromfile(s)
        assert frame6 == frame
        assert frame7 == frame

    def test_frameset(self):
        with vdif.open(SAMPLE_FILE, 'rb') as fh:
            header = vdif.VDIFHeader.fromfile(fh)
            fh.seek(0)
            frameset = fh.read_frameset()

        assert len(frameset.frames) == 8
        assert len(frameset) == len(frameset.frames[0])
        assert frameset.samples_per_frame == 20000
        assert frameset.nchan == 1
        assert frameset.shape == (20000, 8, 1)
        assert frameset.size == 160000
        assert frameset.ndim == 3
        assert frameset.nbytes == 8 * frameset.frames[0].nbytes
        assert 'edv' in frameset
        assert 'edv' in frameset.keys()
        assert frameset['edv'] == 3
        # Properties from headers are passed on if they are settable.
        assert frameset.time == frameset.header0.time
        with pytest.raises(AttributeError):
            # But not all.
            frameset.update(1)
        assert ([fr.header['thread_id'] for fr in frameset.frames] ==
                list(range(8)))
        # Check frame associated with header makes sense.
        first_frame = frameset.frames[header['thread_id']]
        assert first_frame.header == header
        assert np.all(first_frame[:12, 0].astype(int) ==
                      np.array([1, 1, 1, -3, 1, 1, -3, -3, -3, 3, 3, -1]))
        # Check two other frames are in the right place.
        assert np.all(frameset.frames[0][:12, 0].astype(int) ==
                      np.array([-1, -1, 3, -1, 1, -1, 3, -1, 1, 3, -1, 1]))
        assert np.all(frameset.frames[3][:12, 0].astype(int) ==
                      np.array([-1, 1, -1, 1, -3, -1, 3, -1, 3, -3, 1, 3]))

        with vdif.open(SAMPLE_FILE, 'rb') as fh:
            frameset2 = fh.read_frameset(thread_ids=[2, 3])
            fh.fh_raw.seek(0)
            frameset3 = fh.read_frameset(thread_ids=[3, 4, 1])
        assert frameset2.shape == (20000, 2, 1)
        # Note: slicing of framesets themselves is checked below.
        assert np.all(frameset2.data == frameset.data[:, 2:4])
        assert np.all(frameset3.data == frameset.data[:, [3, 4, 1]])

        frameset3 = vdif.VDIFFrameSet(frameset.frames, frameset.header0)
        assert frameset3 == frameset
        frameset4 = vdif.VDIFFrameSet.fromdata(frameset.data, frameset.header0)
        assert np.all(frameset4.data == frameset.data)
        assert frameset4.time == frameset.time
        # the following check cannot be done with frameset itself, since
        # the times in four of the eight headers are wrong.
        frameset5 = vdif.VDIFFrameSet.fromdata(frameset.data,
                                               frameset4.header0)
        assert frameset5 == frameset4
        frameset6 = vdif.VDIFFrameSet.fromdata(frameset.data,
                                               **frameset4.header0)
        assert frameset6 == frameset4

        with vdif.open(SAMPLE_FILE, 'rb') as fh:
            fh.seek(0)
            # try reading just a few threads
            frameset4 = fh.read_frameset(thread_ids=[2, 3])
            assert frameset4.header0.time == frameset.header0.time
            assert np.all(frameset.data[:, 2:4] == frameset4.data)
            # Read beyond end
            fh.seek(-10064, 2)
            with pytest.raises(EOFError):
                fh.read_frameset(thread_ids=list(range(8)))
            # Give non-existent thread_id.
            fh.seek(0)
            with pytest.raises(IOError):
                fh.read_frameset(thread_ids=[1, 9])

    def test_frameset_getitem_setitem(self):
        with vdif.open(SAMPLE_FILE, 'rb') as fh:
            frameset = fh.read_frameset()

        # Try slicing a bit more, assuming whole-frameset read works.
        data = frameset.data
        # Whole frameset.
        assert np.all(frameset[()] == data)
        assert np.all(frameset[:] == data)
        # Select just samples.
        assert np.all(frameset[15] == data[15])
        assert np.all(frameset[10:20] == data[10:20])
        # Select just frames.
        assert np.all(frameset[:, 3] == data[:, 3])
        assert np.all(frameset[:, 2:4] == data[:, 2:4])
        # Select just channels (there is only one).
        assert np.all(frameset[:, :, 0] == data[:, :, 0])
        assert np.all(frameset[:, :, :1] == data[:, :, :1])
        # Check direct access to frames with known results.
        assert np.all(frameset[:12, 0, 0].astype(int) ==
                      np.array([-1, -1, 3, -1, 1, -1, 3, -1, 1, 3, -1, 1]))
        assert np.all(frameset[:12, 3, 0].astype(int) ==
                      np.array([-1, 1, -1, 1, -3, -1, 3, -1, 3, -3, 1, 3]))

        frameset2 = vdif.VDIFFrameSet.fromdata(data, frameset.header0)
        assert np.all(frameset2.data == data)
        frameset2[()] = 1.
        assert np.all(frameset2.data == 1.)
        frameset2[:] = data
        assert np.all(frameset2.data == data)
        # Select just samples.
        frameset2[15] = -data[15]
        assert np.all(frameset2[15] == -data[15])
        frameset2[10:20] = -1.
        assert np.all(frameset2[10:20] == -1.)
        frameset2[10:20:2] = data[10:20:2]
        assert np.all(frameset2[10:20:2] == data[10:20:2])
        assert np.all(frameset2[11:20:2] == -1.)
        # Select just frames.
        frameset2[:, 3] = -1
        assert np.all(frameset2[:, 3] == -1.)
        frameset2[:, 2:4] = 1.
        assert np.all(frameset2[:, 2:4] == 1.)
        frameset2[:, [0, 4, 5, 6]] = data[:, :4]
        frameset2[:, [1, 2, 3, 7]] = data[:, 4:]
        assert np.all(frameset2[:, [0, 4, 5, 6, 1, 2, 3, 7]] == data)
        # Select just channels (there is only one).
        frameset2[:, :, 0] = -data[:, :, 0]
        assert np.all(frameset2[:, :, 0] == -data[:, :, 0])
        frameset2[:, :, :1] = 1.
        assert np.all(frameset2[:, :, :1] == 1.)
        # Test header getting/setting.
        assert np.all(frameset2['thread_id'] == [f.header['thread_id']
                                                 for f in frameset2.frames])
        assert frameset2['frame_nr'] == frameset2.header0['frame_nr']
        frameset2['frame_nr'] = 25
        assert all(f.header['frame_nr'] == 25 for f in frameset2.frames)
        assert frameset2['frame_nr'] == 25
        frameset2['thread_id'] = np.arange(10, 18)
        assert all(f.header['thread_id'] == v
                   for f, v in zip(frameset2.frames, np.arange(10, 18)))
        assert all(frameset2['thread_id'] == np.arange(10, 18))
        with pytest.raises(ValueError):
            frameset2['thread_id'] = 0
        with pytest.raises(ValueError):
            frameset2['thread_id'] = 0, 1, 2, 3, 4, 5, 6, 1
        with pytest.raises(ValueError):
            frameset2['frame_nr'] = 0, 1, 0, 1, 0, 1, 0, 1
        # And a bit of getattr.
        assert frameset2.time == frameset2.header0.time
        assert frameset2.valid
        mixed_valid = True, True, False, False, True, True, False, False
        frameset2.valid = mixed_valid
        assert np.all(frameset2.valid == mixed_valid)
        frameset2.valid = True
        assert frameset2.valid
        frameset2.valid = False
        assert not frameset2.valid

    def test_find_header(self, tmpdir):
        # Below, the tests set the file pointer to very close to a header,
        # since otherwise they run *very* slow.  This is somehow related to
        # pytest, since speed is not a big issue running stuff on its own.
        with vdif.open(SAMPLE_FILE, 'rb') as fh:
            header0 = vdif.VDIFHeader.fromfile(fh)
            fh.seek(0)
            header_0 = fh.find_header(frame_nbytes=header0.frame_nbytes)
            assert fh.tell() == 0
            fh.seek(5000)
            header_5000f = fh.find_header(frame_nbytes=header0.frame_nbytes,
                                          forward=True)
            assert fh.tell() == header0.frame_nbytes
            # sample file has corrupted time in even threads; check this
            # doesn't matter
            fh.seek(15000)
            header_15000f = fh.find_header(frame_nbytes=header0.frame_nbytes,
                                           forward=True)
            assert fh.tell() == 3 * header0.frame_nbytes
            fh.seek(20128)
            header_20128f = fh.find_header(template_header=header0,
                                           forward=True)
            assert fh.tell() == 4 * header0.frame_nbytes
            fh.seek(16)
            header_16b = fh.find_header(frame_nbytes=header0.frame_nbytes,
                                        forward=False)
            assert fh.tell() == 0
            fh.seek(-10000, 2)
            header_m10000b = fh.find_header(frame_nbytes=header0.frame_nbytes,
                                            forward=False)
            assert fh.tell() == 14 * header0.frame_nbytes
            fh.seek(-5000, 2)
            header_m5000b = fh.find_header(frame_nbytes=header0.frame_nbytes,
                                           forward=False)
            assert fh.tell() == 15 * header0.frame_nbytes
            fh.seek(-20, 2)
            header_end = fh.find_header(template_header=header0,
                                        forward=True)
            assert header_end is None
        # thread order = 1,3,5,7,0,2,4,6
        assert header_16b == header_0
        # second frame
        assert header_5000f['frame_nr'] == 0
        assert header_5000f['thread_id'] == 3
        # fourth frame
        assert header_15000f['frame_nr'] == 0
        assert header_15000f['thread_id'] == 7
        # fifth frame
        assert header_20128f['frame_nr'] == 0
        assert header_20128f['thread_id'] == 0
        # one but last frame
        assert header_m10000b['frame_nr'] == 1
        assert header_m10000b['thread_id'] == 4
        # last frame
        assert header_m5000b['frame_nr'] == 1
        assert header_m5000b['thread_id'] == 6
        # Make file with missing data.
        with open(str(tmpdir.join('test.vdif')), 'w+b') as s, \
                open(SAMPLE_FILE, 'rb') as f:
            s.write(f.read(5100))
            f.seek(10000)
            s.write(f.read())
            with vdif.open(s, 'rb') as fh:
                fh.seek(0)
                header_0 = fh.find_header(template_header=header0)
                assert fh.tell() == 0
                fh.seek(5000)
                header_5000ft = fh.find_header(template_header=header0,
                                               forward=True)
                assert fh.tell() == header0.frame_nbytes * 2 - 4900
                header_5000f = fh.find_header(
                    frame_nbytes=header0.frame_nbytes, forward=True)
                assert fh.tell() == header0.frame_nbytes * 2 - 4900
        assert header_5000f['frame_nr'] == 0
        assert header_5000f['thread_id'] == 5
        assert header_5000ft == header_5000f
        # for completeness, also check a really short file...
        with open(str(tmpdir.join('test.vdif')), 'w+b') as s, \
                open(SAMPLE_FILE, 'rb') as f:
            s.write(f.read(5040))
            with vdif.open(s, 'rb') as fh:
                fh.seek(10)
                header_10 = fh.find_header(frame_nbytes=header0.frame_nbytes,
                                           forward=False)
                assert fh.tell() == 0
            assert header_10 == header0

    def test_filestreamer(self):
        with open(SAMPLE_FILE, 'rb') as fh:
            header = vdif.VDIFHeader.fromfile(fh)

        with vdif.open(SAMPLE_FILE, 'rs') as fh:
            assert fh.readable() is True
            assert fh.writable() is False
            assert fh.seekable() is True
            assert fh.closed is False
            assert repr(fh).startswith('<VDIFStreamReader')
            assert fh.tell() == 0
            assert header == fh.header0
            assert fh.sample_rate == 32*u.MHz
            assert fh.start_time == fh.header0.time
            assert abs(fh.time - fh.start_time) < 1. * u.ns
            assert fh.time == fh.tell(unit='time')
            record = fh.read(12)
            assert fh.tell() == 12
            t12 = fh.time
            s12 = 12 / fh.sample_rate
            assert abs(t12 - fh.start_time - s12) < 1. * u.ns
            fh.seek(10, 1)
            fh.tell() == 22
            fh.seek(t12)
            assert fh.tell() == 12
            fh.seek(-s12, 1)
            assert fh.tell() == 0
            with pytest.raises(ValueError):
                fh.seek(0, 3)
            # Test seeker works with both int and str values for whence
            assert fh.seek(13, 0) == fh.seek(13, 'start')
            assert fh.seek(-13, 2) == fh.seek(-13, 'end')
            fhseek_int = fh.seek(17, 1)
            fh.seek(-17, 'current')
            fhseek_str = fh.seek(17, 'current')
            assert fhseek_int == fhseek_str
            with pytest.raises(ValueError):
                fh.seek(0, 'last')
            assert fh.sample_shape == (8,)
            assert fh.shape == (40000,) + fh.sample_shape
            assert fh.size == np.prod(fh.shape)
            assert fh.ndim == len(fh.shape)
            assert abs(fh.stop_time - fh._last_header.time - (
                fh.samples_per_frame / fh.sample_rate)) < 1. * u.ns
            assert abs(fh.stop_time - fh.start_time -
                       (fh.shape[0] / fh.sample_rate)) < 1. * u.ns
            fh.seek(1, 'end')
            with pytest.raises(EOFError):
                fh.read()

        assert record.shape == (12, 8)
        assert np.all(record.astype(int)[:, 0] ==
                      np.array([-1, -1, 3, -1, 1, -1, 3, -1, 1, 3, -1, 1]))

        # Test that squeeze attribute works on read (including in-place read).
        with vdif.open(SAMPLE_FILE, 'rs') as fh:
            assert fh.sample_shape == (8,)
            assert fh.sample_shape.nthread == 8
            assert fh.read(1).shape == (1, 8)
            fh.seek(0)
            out_squeeze = np.zeros((12, 8))
            fh.read(out=out_squeeze)
            assert fh.tell() == 12
            assert np.all(out_squeeze == record)

        with vdif.open(SAMPLE_FILE, 'rs', squeeze=False) as fh:
            assert fh.sample_shape == (8, 1)
            assert fh.read(1).shape == (1, 8, 1)
            fh.seek(0)
            out_nosqueeze = np.zeros((12, 8, 1))
            fh.read(out=out_nosqueeze)
            assert fh.tell() == 12
            assert np.all(out_nosqueeze.squeeze() == out_squeeze)

    def test_stream_writer(self, tmpdir):
        vdif_file = str(tmpdir.join('simple.vdif'))
        # Try writing a very simple file, using edv=0.  With 16 samples per
        # frame and a sample_rate of 320 Hz, we have a frame rate of 20 Hz.
        # We start on purpose not on an integer second, to check that
        # frame numbers get set correctly automatically.
        start_time = Time('2010-11-12T13:14:15.25')
        # Same fake data for all frames.
        data = np.ones((16, 2, 2))
        data[5, 0, 0] = data[6, 1, 1] = -1.
        header = vdif.VDIFHeader.fromvalues(
            edv=0, time=start_time, nchan=2, bps=2,
            complex_data=False, thread_id=0, samples_per_frame=16,
            station='me', sample_rate=320*u.Hz)
        with vdif.open(vdif_file, 'ws', header0=header,
                       sample_rate=320*u.Hz, nthread=2) as fw:
            assert fw.sample_rate == 320*u.Hz
            for i in range(17):
                fw.write(data)
            # Write an invalid frame.
            fw.write(data, valid=False)
            # Write 3 frames using pieces.
            fw.write(data[:4])
            fw.write(np.concatenate((data[4:], data, data[:-4]), axis=0))
            fw.write(data[-4:])
            # Add 9 more just to fill it out to 30 frames.
            for i in range(9):
                fw.write(data)

        with vdif.open(vdif_file, 'rs') as fh:
            assert fh.header0.station == 'me'
            assert fh.samples_per_frame == 16
            assert fh.sample_rate == 320*u.Hz
            assert not fh.complex_data
            assert fh.header0.bps == 2
            assert fh.sample_shape.nchan == 2
            assert fh.sample_shape.nthread == 2
            assert fh.start_time == start_time
            assert np.abs(fh.stop_time - fh.start_time - 1.5 * u.s) < 1. * u.ns
            # Seek to 8 samples before the 17th frame (which has invalid data),
            # and check reading around that.
            fh.seek(16 * 17 - 8)
            record = fh.read(56)
            assert np.all(record[:8] == data[8:])
            assert np.all(record[8:24] == 0.)
            assert np.all(record[24:40] == data)
            assert np.all(record[40:] == data)

        # A bit random, but this is a good place to check that `info`
        # does the right thing for streams that do not start at frame nr 0.
        with vdif.open(vdif_file, 'rb') as fh:
            assert fh.info.frame_rate == 20. * u.Hz
            assert abs(fh.info.start_time - start_time) < 1. * u.ns

        # Test that failing to pass a sample rate returns a ValueError.
        # TODO: why can this not be determined from the frame rate?
        with pytest.raises(ValueError) as excinfo:
            with vdif.open(vdif_file, 'ws', header0=header,
                           nthread=2) as fw:
                pass
        assert "sample rate must be passed" in str(excinfo.value)

        # Test that squeeze attribute works on write.
        with vdif.open(SAMPLE_FILE, 'rs') as fh:
            record = fh.read()
            header = fh.header0

        test_file_squeeze = str(tmpdir.join('test_squeeze.vdif'))
        with vdif.open(test_file_squeeze, 'ws', header0=header,
                       nthread=8) as fws:
            assert fws.sample_shape == (8,)
            assert fws.sample_shape.nthread == 8
            fws.write(record)
        test_file_nosqueeze = str(tmpdir.join('test_nosqueeze.vdif'))
        with vdif.open(test_file_nosqueeze, 'ws', header0=header,
                       nthread=8, squeeze=False) as fwns:
            assert fwns.sample_shape == (8, 1)
            assert fwns.sample_shape.nthread == 8
            assert fwns.sample_shape.nchan == 1
            fwns.write(record[..., np.newaxis])

        with vdif.open(test_file_squeeze, 'rs') as fhs, \
                vdif.open(test_file_nosqueeze, 'rs') as fhns:
            assert np.all(fhs.read() == record)
            assert np.all(fhns.read() == record)

    def test_subset(self, tmpdir):
        # Use the default sample file.
        with vdif.open(SAMPLE_FILE, 'rs') as fh:
            data = fh.read()
            sample_rate = fh.sample_rate
            samples_per_frame = fh.samples_per_frame
            header0 = fh.header0

        with vdif.open(SAMPLE_FILE, 'rs', subset=slice(0, 8, 2)) as fhn:
            assert fhn.sample_shape == (4,)
            check = fhn.read()
            assert np.all(check == data[:, slice(0, 8, 2)])

        with vdif.open(SAMPLE_FILE, 'rs', subset=[0]) as fhn:
            assert fhn.sample_shape == (1,)
            check = fhn.read()
            assert check.shape == (data.shape[0], 1)
            assert np.all(check == data[:, :1])

        # Make an 8 channel file.
        test_file = str(tmpdir.join('test.vdif'))
        with vdif.open(test_file, 'ws', sample_rate=sample_rate,
                       samples_per_frame=samples_per_frame // 8, nthread=1,
                       nchan=8, complex_data=header0['complex_data'],
                       bps=header0.bps, edv=header0.edv,
                       station=header0.station, time=fh.start_time) as fw:
            fw.write(data)

        with vdif.open(test_file, 'rs') as fhn:
            assert np.all(fhn.read() == data)

        with vdif.open(SAMPLE_FILE, 'rs', subset=np.array([3, 7])) as fhn:
            assert fhn.sample_shape == (2,)
            check = fhn.read()
            assert np.all(check == data[:, np.array([3, 7])])

        with vdif.open(SAMPLE_FILE, 'rs', subset=[2]) as fhn:
            assert fhn.sample_shape == (1,)
            check = fhn.read()
            assert np.all(check == data[:, 2:3])

        # Make an 8 thread, 4 channel file.
        data4x = np.array([data, abs(data),
                           -data, -abs(data)]).transpose(1, 2, 0)
        with vdif.open(test_file, 'ws', sample_rate=sample_rate,
                       samples_per_frame=samples_per_frame // 4, nthread=8,
                       nchan=4, complex_data=header0['complex_data'],
                       bps=header0.bps, edv=header0.edv,
                       station=header0.station, time=fh.start_time) as fw:
            fw.write(data4x)

        # Sanity check by re-reading the file.
        with vdif.open(test_file, 'rs') as fhn:
            assert fhn.sample_shape == (8, 4)
            check = fhn.read()
            assert np.all(check == data4x)

        # Single thread and channel selection.
        with vdif.open(test_file, 'rs', subset=(6, 2)) as fhn:
            assert fhn.sample_shape == ()
            check = fhn.read()
            assert np.all(check == data4x[:, 6, 2])

        # Single thread, multi-channel selection.
        with vdif.open(test_file, 'rs', subset=(3, [1, 2])) as fhn:
            assert fhn.sample_shape == (2,)
            check = fhn.read()
            assert np.all(check == data4x[:, 3, 1:3])

        # Multi-thread, multi-channel selection
        subset_md = (np.array([5, 3])[:, np.newaxis], np.array([0, 2]))
        with vdif.open(test_file, 'rs', subset=subset_md) as fhn:
            assert fhn.sample_shape == (2, 2)
            thread_ids = [frame.header['thread_id'] for frame in
                          fhn._frameset.frames]
            assert thread_ids == [5, 3]
            check = fhn.read()
            assert np.all(check == data4x[(slice(None),) + subset_md])

    def test_stream_verify(self, tmpdir):
        """Test that we can pass or set verify=False to bypass header
        checking."""
        # Grab data from the sample file.
        with vdif.open(SAMPLE_FILE, 'rs') as fh:
            data = fh.read()

        testverifyfile = str(tmpdir.join('testverify.vdif'))
        # Make a file with a sync pattern error in the second set of frames.
        with vdif.open(SAMPLE_FILE, 'rb') as fh, \
                vdif.open(testverifyfile, 'wb') as fw:
            fr = fh.read_frameset()
            fr.tofile(fw)
            fr = fh.read_frameset()
            fr.frames[2].header.mutable = True
            fr.frames[2].header['sync_pattern'] = 0xabbaabba
            fr.tofile(fw)

        with vdif.open(testverifyfile, 'rs') as fn:
            assert fn.verify is True
            # This should fail at the second frameset.
            with pytest.raises(AssertionError):
                fn.read()
            assert fn.tell() == 20000
            fn.verify = False
            assert fn.verify is False
            assert np.all(fn.read() == data[20000:])

        # Check that we can pass verify=False.
        with vdif.open(testverifyfile, 'rs', verify=False) as fn:
            assert fn.verify is False
            assert np.all(fn.read() == data)

    # Test that writing an incomplete stream is possible, and that frame set is
    # appropriately marked as invalid.
    @pytest.mark.parametrize('fill_value', (0., -999.))
    def test_incomplete_stream(self, tmpdir, fill_value):
        vdif_incomplete = str(tmpdir.join('incomplete.vdif'))
        with catch_warnings(UserWarning) as w:
            with vdif.open(SAMPLE_FILE, 'rs') as fr:
                record = fr.read(10)
                with vdif.open(vdif_incomplete, 'ws', header0=fr.header0,
                               sample_rate=32*u.MHz, nthread=8) as fw:
                    fw.write(record)
        assert len(w) == 1
        assert 'partial buffer' in str(w[0].message)
        with vdif.open(vdif_incomplete, 'rs', fill_value=fill_value) as fwr:
            assert all([not frame.valid for frame in fwr._frameset.frames])
            assert fwr.fill_value == fill_value
            assert np.all(fwr.read() == fill_value)

    def test_corrupt_stream(self, tmpdir):
        with vdif.open(SAMPLE_FILE, 'rb') as fh, \
                open(str(tmpdir.join('test.vdif')), 'w+b') as s:
            frame = fh.read_frame()
            frame.tofile(s)
            # Now add lots of the next frame, i.e., with a different thread_id
            # and different frame_nr
            fh.seek(-frame.header.frame_nbytes, 2)
            frame2 = fh.read_frame()
            for i in range(15):
                frame2.tofile(s)
            s.seek(0)
            with vdif.open(s, 'rs') as f2:
                assert f2.header0 == frame.header
                with pytest.raises(ValueError):
                    f2._last_header

    def test_io_invalid(self, tmpdir):
        tmp_file = str(tmpdir.join('ts.dat'))
        with open(tmp_file, 'wb') as fw:
            fw.write(b'      ')

        with pytest.raises(TypeError):
            # Extra argument.
            vdif.open(tmp_file, 'rb', bla=10)
        with pytest.raises(ValueError):
            # Missing w or r.
            vdif.open(tmp_file, 's')


def test_sequentialfile(tmpdir):
    """Tests writing and reading of sequential files.

    These tests focus on reading and writing with templates.
    """

    # Use sample file as basis of a file sequence.
    with vdif.open(SAMPLE_FILE, 'rs') as fh:
        header = fh.header0.copy()
        data = fh.read()
        dtime = fh.stop_time - fh.start_time
    data = np.concatenate((data, data, data))

    # Create a file sequence using template.
    template = str(tmpdir.join('f.{file_nr:03d}.vdif'))
    files = sf.FileNameSequencer(template)
    with vdif.open(files, 'ws', file_size=16*header.frame_nbytes,
                   nthread=8, **header) as fw:
        fw.write(data)

    # Read in file-sequence and check data consistency.
    with vdif.open(files, 'rs') as fn:
        assert len(fn.fh_raw.files) == 3
        assert fn.fh_raw.files[-1] == str(tmpdir.join('f.002.vdif'))
        assert fn.header0.time == header.time
        assert fn.stop_time - fn.start_time - 3 * dtime < 1 * u.ns
        assert np.all(data == fn.read())

    # Read in one file and check if everything makes sense.
    with vdif.open(template.format(file_nr=2), 'rs') as fn:
        assert fn.header0.time - header.time - 2. * dtime < 1 * u.ns
        assert np.all(data[80000:] == fn.read())


def test_vlbi_vdif():
    """Tests SAMPLE_VLBI, which is SAMPLE_VDIF with uncorrected timestamps."""
    with vdif.open(SAMPLE_VLBI, 'rs') as fh, \
            vdif.open(SAMPLE_FILE, 'rs') as fhc:
        assert fh.sample_rate == 32*u.MHz
        assert fh.start_time == fh.header0.time
        assert fh.start_time == fhc.start_time
        assert fh.shape == (40000,) + fh.sample_shape
        assert abs(fh.stop_time - fh._last_header.time - (
            fh.samples_per_frame / fh.sample_rate)) < 1. * u.ns
        assert abs(fh.stop_time - fh.start_time -
                   (fh.shape[0] / fh.sample_rate)) < 1. * u.ns
        assert np.all(fh.read() == fhc.read())


def test_mwa_vdif():
    """Test phased VDIF format (uses EDV=0)"""
    with vdif.open(SAMPLE_MWA, 'rs', sample_rate=1.28*u.MHz) as fh:
        assert fh.samples_per_frame == 128
        assert fh.sample_rate == 1.28*u.MHz
        assert fh.time == Time('2015-10-03T20:49:45.000')
        assert fh.header0.edv == 0


def test_arochime_vdif():
    """Test ARO CHIME format (uses EDV=0)"""
    # File has 1 sample/frame.
    frame_rate = sample_rate = 800*u.MHz / 1024. / 2.
    with open(SAMPLE_AROCHIME, 'rb') as fh:
        header0 = vdif.VDIFHeader.fromfile(fh)
    assert header0.edv == 0
    assert header0.samples_per_frame == 1
    assert header0['frame_nr'] == 308109
    with pytest.raises(ValueError):
        header0.time
    assert abs(header0.get_time(frame_rate=frame_rate) -
               Time('2016-04-22T08:45:31.788759040')) < 1. * u.ns
    # Also check writing Time.
    header1 = header0.copy()
    with pytest.raises(ValueError):
        header1.time = Time('2016-04-22T08:45:31.788759040')
    header1.set_time(Time('2016-04-22T08:45:32.788759040'),
                     frame_rate=frame_rate)
    assert abs(header1.get_time(frame_rate=frame_rate) -
               header0.get_time(frame_rate=frame_rate) - 1. * u.s) < 1.*u.ns

    # Now test the actual data stream.
    with vdif.open(SAMPLE_AROCHIME, 'rs', sample_rate=sample_rate) as fh:
        assert fh.samples_per_frame == 1
        t0 = fh.time
        assert abs(t0 - Time('2016-04-22T08:45:31.788759040')) < 1. * u.ns
        assert abs(t0 - fh.start_time) < 1. * u.ns
        assert fh.header0.edv == 0
        assert fh.shape == (5,) + fh.sample_shape
        d = fh.read()
        assert d.shape == (5, 2, 1024)
        assert d.dtype.kind == 'c'
        t1 = fh.time
        assert abs(t1 - fh.stop_time) < 1. * u.ns
        assert abs(t1 - t0 - fh.shape[0] / fh.sample_rate) < 1. * u.ns

    # For this file, we cannot find a frame rate, so opening it without
    # should fail.
    with pytest.raises(EOFError):
        with vdif.open(SAMPLE_AROCHIME, 'rs') as fh:
            pass


def test_legacy_vdif(tmpdir):
    """Create legacy header, ensuring it is not treated as EDV=0 (see #12)."""
    # legacy_mode, 1 sec, epoch 4, vdif v1, nchan=2, 507*8 bytes, bps=2, 'AA'
    words = (1 << 30 | 1, 4 << 24, 1 << 29 | 1 << 24 | 507, 1 << 26 | 0x4141)
    header = vdif.VDIFHeader(words)
    assert header['legacy_mode'] is True
    assert header.edv is False
    assert abs(header.time - Time('2002-01-01T00:00:01.000000000')) < 1. * u.ns
    assert header['frame_nr'] == 0
    assert header['vdif_version'] == 1
    assert header.nchan == 2
    assert header.frame_nbytes == 507 * 8
    assert header.nbytes == 16
    assert header['complex_data'] is False
    assert header.bps == 2
    assert header['thread_id'] == 0
    assert header.station == 'AA'
    with open(str(tmpdir.join('test.vdif')), 'w+b') as s:
        header.tofile(s)
        # Add fake payload
        s.write(np.zeros(503, dtype=np.int64).tostring())
        s.seek(0)
        header2 = vdif.VDIFHeader.fromfile(s)
    assert header2 == header
