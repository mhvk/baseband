# Licensed under the GPLv3 - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import pytest
from astropy.time import Time
import astropy.units as u
from astropy.tests.helper import catch_warnings

from ... import vdif, vlbi_base
from ...data import (SAMPLE_VDIF as SAMPLE_FILE, SAMPLE_MWA_VDIF as SAMPLE_MWA,
                     SAMPLE_AROCHIME_VDIF as SAMPLE_AROCHIME)


# Comparisn with m5access routines (check code on 2015-MAY-30) on vlba.m5a,
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
        assert header.size == 32
        assert header.edv == 3
        mjd, frac = divmod(header.time.mjd, 1)
        assert mjd == 56824
        assert round(frac * 86400) == 21367
        assert header.payloadsize == 5000
        assert header.framesize == 5032
        assert header['thread_id'] == 1
        assert header.framerate.value == 1600
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
            edv=header.edv, time=header.time,
            samples_per_frame=header.samples_per_frame,
            station=header.station, bandwidth=header.bandwidth,
            bps=header.bps, complex_data=header['complex_data'],
            thread_id=header['thread_id'],
            loif_tuning=header['loif_tuning'], dbe_unit=header['dbe_unit'],
            if_nr=header['if_nr'], subband=header['subband'],
            sideband=header['sideband'], major_rev=header['major_rev'],
            minor_rev=header['minor_rev'], personality=header['personality'],
            _7_28_4=header['_7_28_4'])
        assert header4 == header
        assert header4.mutable is True
        header5 = header.copy()
        assert header5 == header
        assert header5.mutable is True
        header5['thread_id'] = header['thread_id'] + 1
        assert header5['thread_id'] == header['thread_id'] + 1
        assert header5 != header
        with pytest.raises(TypeError):
            header['thread_id'] = 0
        # Also test time setting
        header5.time = header.time + 1.*u.s
        assert abs(header5.time - header.time - 1.*u.s) < 1.*u.ns
        assert header5['frame_nr'] == header['frame_nr']
        header5.time = header.time + 1.*u.s + 1.1/header.framerate
        assert abs(header5.time - header.time - 1.*u.s -
                   1./header.framerate) < 1.*u.ns
        assert header5['frame_nr'] == header['frame_nr'] + 1
        # Check rounding in corner case.
        header5.time = header.time + 1.*u.s - 0.01/header.framerate
        assert abs(header5.time - header.time - 1.*u.s) < 1.*u.ns
        assert header5['frame_nr'] == header['frame_nr']
        # Check requesting non-existent EDV returns VDIFBaseHeader instance
        header6 = vdif.header.VDIFHeader.fromvalues(edv=100)
        assert isinstance(header6, vdif.header.VDIFBaseHeader)
        assert header6['edv'] == 100

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
            station=header.station, bandwidth=header.bandwidth,
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
        assert headerX.size == header.size
        assert headerX.edv == 0x58
        mjd, frac = divmod(header.time.mjd, 1)
        assert mjd == 56824
        assert round(frac * 86400) == 21367
        assert headerX.payloadsize == header.payloadsize
        assert headerX.framesize == header.framesize
        assert headerX['thread_id'] == header['thread_id']
        assert headerX.framerate.value == header.framerate.value
        assert headerX.samples_per_frame == header.samples_per_frame
        assert headerX.nchan == header.nchan
        assert headerX.bps == header.bps
        assert not headerX['complex_data']
        assert headerX.mutable is False
        assert headerX.nonsense_0 == 2000000000
        assert headerX.nonsense_1 == 100
        assert headerX.nonsense_2 == 10000000

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
                                            payloadsize=payload1.size)
        assert np.all(vdif.VDIFPayload.fromdata(areal, header) == payload1)
        header['complex_data'] = True
        assert np.all(vdif.VDIFPayload.fromdata(acmplx, header) == payload2)
        # Also check for bps=4
        decode = (aint[:, np.newaxis] >> np.array([0, 4])) & 0xf
        areal = ((decode - 8.) / 2.95).reshape(-1, 1)
        acmplx = areal[::2] + 1j * areal[1::2]
        payload3 = vdif.VDIFPayload(words, bps=4, complex_data=False)
        assert np.allclose(payload3.data, areal)
        payload4 = vdif.VDIFPayload(words, bps=4, complex_data=True)
        assert np.allclose(payload4.data, acmplx)
        header = vdif.VDIFHeader.fromvalues(edv=0, complex_data=False, bps=4,
                                            payloadsize=payload3.size)
        assert vdif.VDIFPayload.fromdata(areal, header) == payload3
        header['complex_data'] = True
        assert vdif.VDIFPayload.fromdata(acmplx, header) == payload4

    def test_payload(self, tmpdir):
        with open(SAMPLE_FILE, 'rb') as fh:
            header = vdif.VDIFHeader.fromfile(fh)
            payload = vdif.VDIFPayload.fromfile(fh, header)
        assert payload.size == 5000
        assert payload.shape == (20000, 1)
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
        assert payload4.size == 5000
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

    def test_frame(self, tmpdir):
        with vdif.open(SAMPLE_FILE, 'rb') as fh:
            header = vdif.VDIFHeader.fromfile(fh)
            payload = vdif.VDIFPayload.fromfile(fh, header)
            fh.seek(0)
            frame = fh.read_frame()

        assert frame.header == header
        assert frame.payload == payload
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
        assert frameset.samples_per_frame == 20000
        assert frameset.nchan == 1
        assert frameset.shape == (8, 20000, 1)
        assert frameset.size == 8 * frameset.frames[0].size
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
        first_frame = frameset.frames[header['thread_id']]
        assert first_frame.header == header
        assert np.all(first_frame.data[:12, 0].astype(int) ==
                      np.array([1, 1, 1, -3, 1, 1, -3, -3, -3, 3, 3, -1]))
        assert np.all(frameset.frames[0].data[:12, 0].astype(int) ==
                      np.array([-1, -1, 3, -1, 1, -1, 3, -1, 1, 3, -1, 1]))
        assert np.all(frameset.frames[3].data[:12, 0].astype(int) ==
                      np.array([-1, 1, -1, 1, -3, -1, 3, -1, 3, -3, 1, 3]))

        with vdif.open(SAMPLE_FILE, 'rb') as fh:
            frameset2 = fh.read_frameset(thread_ids=[2, 3])
        assert frameset2.shape == (2, 20000, 1)
        assert np.all(frameset2.data == frameset.data[2:4])

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
            assert np.all(frameset.data[2:4] == frameset4.data)
            # Read beyond end
            fh.seek(-10064, 2)
            with pytest.raises(EOFError):
                fh.read_frameset(thread_ids=list(range(8)))
            # Give non-existent thread_id
            fh.seek(0)
            with pytest.raises(IOError):
                fh.read_frameset(thread_ids=[1, 9])

    def test_find_header(self, tmpdir):
        # Below, the tests set the file pointer to very close to a header,
        # since otherwise they run *very* slow.  This is somehow related to
        # pytest, since speed is not a big issue running stuff on its own.
        with vdif.open(SAMPLE_FILE, 'rb') as fh:
            header0 = vdif.VDIFHeader.fromfile(fh)
            fh.seek(0)
            header_0 = fh.find_header(framesize=header0.framesize)
            assert fh.tell() == 0
            fh.seek(5000)
            header_5000f = fh.find_header(framesize=header0.framesize,
                                          forward=True)
            assert fh.tell() == header0.framesize
            # sample file has corrupted time in even threads; check this
            # doesn't matter
            fh.seek(15000)
            header_15000f = fh.find_header(framesize=header0.framesize,
                                           forward=True)
            assert fh.tell() == 3 * header0.framesize
            fh.seek(20128)
            header_20128f = fh.find_header(template_header=header0,
                                           forward=True)
            assert fh.tell() == 4 * header0.framesize
            fh.seek(16)
            header_16b = fh.find_header(framesize=header0.framesize,
                                        forward=False)
            assert fh.tell() == 0
            fh.seek(-10000, 2)
            header_m10000b = fh.find_header(framesize=header0.framesize,
                                            forward=False)
            assert fh.tell() == 14 * header0.framesize
            fh.seek(-5000, 2)
            header_m5000b = fh.find_header(framesize=header0.framesize,
                                           forward=False)
            assert fh.tell() == 15 * header0.framesize
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
                assert fh.tell() == header0.framesize * 2 - 4900
                header_5000f = fh.find_header(framesize=header0.framesize,
                                              forward=True)
                assert fh.tell() == header0.framesize * 2 - 4900
        assert header_5000f['frame_nr'] == 0
        assert header_5000f['thread_id'] == 5
        assert header_5000ft == header_5000f
        # for completeness, also check a really short file...
        with open(str(tmpdir.join('test.vdif')), 'w+b') as s, \
                open(SAMPLE_FILE, 'rb') as f:
            s.write(f.read(5040))
            with vdif.open(s, 'rb') as fh:
                fh.seek(10)
                header_10 = fh.find_header(framesize=header0.framesize,
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
            assert fh.time_start == fh.header0.time
            assert abs(fh.tell(unit='time') - fh.time_start) < 1. * u.ns
            record = fh.read(12)
            assert fh.tell() == 12
            t12 = fh.tell(unit='time')
            s12 = 12 / fh.samples_per_frame / fh.frames_per_second * u.s
            assert abs(t12 - fh.time_start - s12) < 1. * u.ns
            fh.seek(10, 1)
            fh.tell() == 22
            fh.seek(t12)
            assert fh.tell() == 12
            fh.seek(-s12, 1)
            assert fh.tell() == 0
            with pytest.raises(ValueError):
                fh.seek(0, 3)
            assert fh.size == 40000
            assert abs(fh.time_end - fh._header_last.time - u.s /
                       fh.frames_per_second) < 1. * u.ns
            assert abs(fh.time_end - fh.time_start - u.s * fh.size /
                       fh.samples_per_frame / fh.frames_per_second) < 1. * u.ns

        assert record.shape == (12, 8)
        assert np.all(record.astype(int)[:, 0] ==
                      np.array([-1, -1, 3, -1, 1, -1, 3, -1, 1, 3, -1, 1]))

        # Test that squeeze attribute works on read (including in-place read)
        # but can be turned off if needed.
        with vdif.open(SAMPLE_FILE, 'rs') as fh:
            assert fh.sample_shape == (8,)
            assert fh.sample_shape.nthread == 8
            assert fh.read(1).shape == (8,)
            fh.seek(0)
            out = np.zeros((12, 8))
            fh.read(out=out)
            assert fh.tell() == 12
            assert np.all(out.squeeze() == record)
            fh.squeeze = False
            assert fh.sample_shape == (8, 1)
            assert fh.read(1).shape == (1, 8, 1)
            fh.seek(0)
            out = np.zeros((12, 8, 1))
            fh.read(out=out)
            assert fh.tell() == 12
            assert np.all(out.squeeze() == record)

    def test_stream_writer(self, tmpdir):
        vdif_file = str(tmpdir.join('simple.vdif'))
        # try writing a very simple file, using edv=0
        data = np.ones((16, 2, 2))
        data[5, 0, 0] = data[6, 1, 1] = -1.
        header = vdif.VDIFHeader.fromvalues(
            edv=0, time=Time('2010-01-01'), nchan=2, bps=2,
            complex_data=False, frame_nr=0, thread_id=0, samples_per_frame=16,
            station='me')
        with vdif.open(vdif_file, 'ws', header=header,
                       nthread=2, frames_per_second=20) as fw:
            for i in range(30):
                fw.write(data)

        with vdif.open(vdif_file, 'rs') as fh:
            assert fh.header0.station == 'me'
            assert fh.frames_per_second == 20
            assert fh.samples_per_frame == 16
            assert not fh.complex_data
            assert fh.header0.bps == 2
            assert fh._sample_shape.nchan == 2
            assert fh._sample_shape.nthread == 2
            assert fh.time_start == Time('2010-01-01')
            assert fh.time_end == fh.time_start + 1.5 * u.s
            fh.seek(16)
            record = fh.read(16)
        assert np.all(record == data)

        # Test that squeeze attribute works on write but can be turned off if
        # needed.
        with vdif.open(SAMPLE_FILE, 'rs') as fh:
            record = fh.read()
            header = fh.header0

        test_file = str(tmpdir.join('test.vdif'))
        with vdif.open(test_file, 'ws', nthread=8, header=header) as fw:
            assert fw.sample_shape == (8,)
            assert fw.sample_shape.nthread == 8
            fw.write(record[:20000])
            fw.squeeze = False
            assert fw.sample_shape == (8, 1)
            assert fw.sample_shape.nthread == 8
            assert fw.sample_shape.nchan == 1
            fw.write(record[20000:, :, np.newaxis])

        with vdif.open(test_file, 'rs') as fh:
            assert np.all(fh.read() == record)

    # Test that writing an incomplete stream is possible, and that frame set is
    # appropriately marked as invalid.
    def test_incomplete_stream(self, tmpdir):
        vdif_incomplete = str(tmpdir.join('incomplete.vdif'))
        with catch_warnings(UserWarning) as w:
            with vdif.open(SAMPLE_FILE, 'rs') as fr:
                record = fr.read(10)
                with vdif.open(vdif_incomplete, 'ws', header=fr.header0,
                               nthread=8, frames_per_second=1600) as fw:
                    fw.write(record)
        assert len(w) == 1
        assert 'partial buffer' in str(w[0].message)
        with vdif.open(vdif_incomplete, 'rs') as fwr:
            assert all([not frame.valid for frame in fwr._frameset.frames])
            assert np.all(fwr.read() ==
                          fwr._frameset.invalid_data_value)

    def test_corrupt_stream(self, tmpdir):
        with vdif.open(SAMPLE_FILE, 'rb') as fh, \
                open(str(tmpdir.join('test.vdif')), 'w+b') as s:
            frame = fh.read_frame()
            frame.tofile(s)
            # now add lots of the next frame, i.e., with a different thread_id
            # and different frame_nr
            fh.seek(-frame.header.framesize, 2)
            frame2 = fh.read_frame()
            for i in range(15):
                frame2.tofile(s)
            s.seek(0)
            with vdif.open(s, 'rs') as f2:
                assert f2.header0 == frame.header
                with pytest.raises(ValueError):
                    f2._header_last

    def test_io_invalid(self):
        with pytest.raises(TypeError):
            # extra argument
            vdif.open('ts.dat', 'rb', bla=10)
        with pytest.raises(ValueError):
            # missing w or r
            vdif.open('ts.dat', 's')


def test_mwa_vdif():
    """Test phased VDIF format (uses EDV=0)"""
    with vdif.open(SAMPLE_MWA, 'rs', sample_rate=1.28*u.MHz) as fh:
        assert fh.samples_per_frame == 128
        assert fh.frames_per_second == 10000
        assert fh.tell(unit='time') == Time('2015-10-03T20:49:45.000')
        assert fh.header0.edv == 0


def test_arochime_vdif():
    """Test ARO CHIME format (uses EDV=0)"""
    with open(SAMPLE_AROCHIME, 'rb') as fh:
        header0 = vdif.VDIFHeader.fromfile(fh)
    assert header0.edv == 0
    assert header0.samples_per_frame == 1
    assert header0['frame_nr'] == 308109
    with pytest.raises(ValueError):
        header0.time
    assert abs(header0.get_time(framerate=390625*u.Hz) -
               Time('2016-04-22T08:45:31.788759040')) < 1. * u.ns
    # also check writing Time.
    header1 = header0.copy()
    with pytest.raises(ValueError):
        header1.time = Time('2016-04-22T08:45:31.788759040')
    header1.set_time(Time('2016-04-22T08:45:32.788759040'),
                     framerate=0.390625*u.MHz)
    assert abs(header1.get_time(framerate=390625*u.Hz) -
               header0.get_time(framerate=390625*u.Hz) - 1. * u.s) < 1.*u.ns

    # Now test the actual data stream.
    with vdif.open(SAMPLE_AROCHIME, 'rs', frames_per_second=390625) as fh:
        assert fh.samples_per_frame == 1
        t0 = fh.tell(unit='time')
        assert abs(t0 - Time('2016-04-22T08:45:31.788759040')) < 1. * u.ns
        assert abs(t0 - fh.time_start) < 1. * u.ns
        assert fh.header0.edv == 0
        assert fh.size == 5
        d = fh.read()
        assert d.shape == (5, 2, 1024)
        assert d.dtype.kind == 'c'
        t1 = fh.tell(unit='time')
        assert abs(t1 - fh.time_end) < 1. * u.ns
        assert abs(t1 - t0 - u.s * (fh.size / fh.samples_per_frame /
                                    fh.frames_per_second)) < 1. * u.ns

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
    assert header.framesize == 507 * 8
    assert header.size == 16
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
