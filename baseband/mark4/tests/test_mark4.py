# Licensed under the GPLv3 - see LICENSE
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.tests.helper import catch_warnings
from ... import mark4
from ...vlbi_base.encoding import OPTIMAL_2BIT_HIGH
from ..header import Mark4TrackHeader
from ..payload import reorder32, reorder64
from ...data import (SAMPLE_MARK4 as SAMPLE_FILE,
                     SAMPLE_MARK4_32TRACK as SAMPLE_32TRACK,
                     SAMPLE_MARK4_32TRACK_FANOUT2 as SAMPLE_32TRACK_FANOUT2,
                     SAMPLE_MARK4_16TRACK as SAMPLE_16TRACK)

# Results from mark5access on 2015-JAN-22.
# m5d evn/Ar/gp052d_ar_no0021 MKIV1_4-512-8-2 1000
# Mark5 stream: 0x1a54140
#   stream = File-1/1=evn/gp052d_ar_no0021
#   format = MKIV1_4-512-8-2/1 = 1
#   start mjd/sec = 53171 27492.475000000
#   frame duration = 2500000.00 ns
#   framenum = 0
#   sample rate = 32000000 Hz
#   offset = 3208
#   framebytes = 160000 bytes
#   datasize = 160000 bytes
#   sample granularity = 4
#   frame granularity = 1
#   gframens = 2500000
#   payload offset = -512
#   read position = 0
#   data window size = 1048576 bytes
#   ...
# data at 17, nonzero at line 657 -> item 640.
# Initially this seemed strange, since PAYLOAD_NBITS=20000 leads to 80000
# elements, so one would have expected VALIDSTART*4=96*4=384.
# But the mark5 code has PAYLOAD_OFFSET=(VALIDEND-20000)*f->ntrack/8 = 64*8
# Since each sample takes 2 bytes, one thus expects 384+64*8/2=640. OK.
# So, lines 639--641:
#  0  0  0  0  0  0  0  0
# -1  1  1 -3 -3 -3  1 -1
#  1  1 -3  1  1 -3 -1 -1


class TestMark4(object):
    def test_header_stream(self):
        with open(SAMPLE_FILE, 'rb') as fh:
            fh.seek(0xa88)
            stream = np.fromfile(fh, dtype='<u8', count=5 * 32)
        # check sync words in right place
        assert np.all(stream[64:80] == 0xffffffffffffffff)
        assert mark4.header.crc12.check(stream)
        assert np.all(mark4.header.crc12(stream[:-12]) == stream[-12:])
        words = mark4.header.stream2words(stream)
        assert np.all(mark4.header.words2stream(words) == stream)

    def test_header(self, tmpdir):
        with open(SAMPLE_FILE, 'rb') as fh:
            fh.seek(0xa88)
            header = mark4.Mark4Header.fromfile(fh, ntrack=64, decade=2010)
            fh.seek(-10, 2)
            with pytest.raises(EOFError):
                mark4.Mark4Header.fromfile(fh, ntrack=64, decade=2010)

        assert len(header) == 64
        assert header.track_id[0] == 2 and header.track_id[-1] == 33
        assert header.nbytes == 160 * 64 // 8
        assert header.fanout == 4
        assert header.bps == 2
        assert not np.all(~header['magnitude_bit'])
        assert header.nchan == 8
        assert int(header.time.mjd) == 56824
        assert header.time.isot == '2014-06-16T07:38:12.47500'
        assert header.samples_per_frame == 20000 * 4
        assert header.frame_nbytes == 20000 * 64 // 8
        assert header.payload_nbytes == header.frame_nbytes - header.nbytes
        assert header.mutable is False
        assert header.nsb == 1
        assert np.all(header.converters['converter'] ==
                      [0, 2, 1, 3, 4, 6, 5, 7])
        assert np.all(header.converters['lsb'])
        assert repr(header).startswith('<Mark4Header bcd_headstack1: [0')
        with open(str(tmpdir.join('test.m4')), 'w+b') as s:
            header.tofile(s)
            s.seek(0)
            header2 = mark4.Mark4Header.fromfile(s, header.ntrack,
                                                 header.decade)
        assert header2 == header
        assert header2.mutable is False
        header3 = mark4.Mark4Header.fromkeys(header.ntrack, header.decade,
                                             **header)
        assert header3 == header
        assert header3.mutable is True
        # Try initialising with properties instead of keywords.
        # Here, we let time imply the decade, bcd_unit_year, bcd_day, bcd_hour,
        # bcd_minute, bcd_second, bcd_fraction;
        # and ntrack, samples_per_frame, bps define headstack_id, bcd_track_id,
        # fan_out, magnitude_bit, and converter_id.
        header4 = mark4.Mark4Header.fromvalues(
            ntrack=64, samples_per_frame=80000, bps=2, nsb=1, time=header.time,
            system_id=108)
        assert header4 == header
        assert header4.mutable is True
        # Check that passing a year into decade leads to an error.
        with pytest.raises(AssertionError):
            mark4.Mark4Header(header.words, decade=2014)
        # Check that passing approximate ref_time is equivalent to passing a
        # decade.
        with open(SAMPLE_FILE, 'rb') as fh:
            fh.seek(0xa88)
            header5 = mark4.Mark4Header.fromfile(
                fh, ntrack=64, ref_time=Time('2010:351:12:00:00.0'))
            assert header5 == header
            fh.seek(0xa88)
            header6 = mark4.Mark4Header.fromfile(
                fh, ntrack=64, ref_time=Time('2018:113:16:30:00.0'))
            assert header6 == header
        # Check changing properties.
        header7 = header.copy()
        assert header7 == header
        assert header7.mutable is True
        header7['bcd_headstack1'] = 0x5566
        assert np.all(header7['bcd_headstack1'] == 0x5566)
        assert header7 != header
        header7['bcd_headstack1'] = np.hstack((0x7788,
                                               header7['bcd_headstack1'][1:]))
        assert header7['bcd_headstack1'][0] == 0x7788
        assert np.all(header7['bcd_headstack1'][1:] == 0x5566)
        # Check it doesn't work on non-mutable header
        with pytest.raises(TypeError):
            header['bcd_headstack1'] = 0
        # Check time assignment
        with pytest.raises(ValueError):
            header7.time = header.time + 0.1 * u.ms
        header7.bps = 1
        assert np.all(~header7['magnitude_bit'])
        header7.bps = 2
        assert not np.all(~header7['magnitude_bit'])
        with pytest.raises(ValueError):
            header7.bps = 4
        with pytest.raises(AttributeError):
            header7.ntrack = 51
        with pytest.raises(AttributeError):
            header7.frame_nbytes = header.frame_nbytes
        with pytest.raises(AttributeError):
            header7.payload_nbytes = header.payload_nbytes
        header7.nchan = 16
        assert header7.nchan == 16 and header7.bps == 1
        # OK, this is silly, but why not...
        header7.time = header.time + np.arange(64) * 125 * u.ms
        assert len(header7.time) == 64
        assert np.all(abs(header7.time - header.time -
                          np.arange(64) * 125 * u.ms) < 1.*u.ns)
        with pytest.raises(ValueError):  # different decades
            header7.time = header.time - (np.linspace(0, 20, 64) // 4) * 4*u.yr
        # Check slicing.
        header8 = header[:2]
        assert type(header8) is mark4.Mark4Header
        assert len(header8) == 2
        header9 = header[10]
        assert type(header9) is Mark4TrackHeader
        assert repr(header9).startswith('<Mark4TrackHeader bcd_headstack1: 0')
        assert repr(header[:1]).startswith('<Mark4Header bcd_headstack1: 0')
        header10 = Mark4TrackHeader(header9.words, decade=2010)
        assert header10 == header9
        header11 = Mark4TrackHeader.fromvalues(decade=2010, **header9)
        assert header11 == header9
        # Check passing decade=None still reads the header, and we can set
        # decade afterward.
        with open(SAMPLE_FILE, 'rb') as fh:
            fh.seek(0xa88)
            header12 = mark4.Mark4Header.fromfile(fh, ntrack=64)
            assert header12.decade is None
            header12.decade = 2010
            assert header12 == header
        with pytest.raises(IndexError):
            header[65]
        with pytest.raises(ValueError):
            header[np.array([[0, 1], [2, 3]])]
        # Check that one can construct crazy headers, even if not much works.
        header13 = mark4.Mark4Header(None, ntrack=53, decade=2010,
                                     verify=False)
        header13.time = header.time
        assert header13.ntrack == 53
        assert abs(header13.time - header.time) < 1. * u.ns

    @pytest.mark.parametrize(('unit_year', 'ref_time', 'decade'),
                             [(5, Time('2014:1:12:00:00'), 2010),
                              (5, Time('2009:362:19:27:33'), 2000),
                              (4, Time('2009:001:19:27:33'), 2010),
                              (3, Time('2018:117:6:42:15'), 2020),
                              (4, Time('2018:117:6:42:15'), 2010)])
    def test_infer_decade(self, unit_year, ref_time, decade):
        # Check that infer_decade returns proper decade for
        # ref_time.year - 5 < year < ref_time.year + 5, and uses bankers'
        # rounding at the boundaries.
        header = mark4.header.Mark4Header(None, ntrack=16, verify=False)
        header['bcd_unit_year'] = unit_year
        header.infer_decade(ref_time)
        assert header.decade == decade

    def test_decoding(self):
        """Check that look-up levels are consistent with mark5access."""
        o2h = OPTIMAL_2BIT_HIGH
        assert np.all(mark4.payload.lut1bit[0] == 1.)
        assert np.all(mark4.payload.lut1bit[0xff] == -1.)
        assert np.all(mark4.payload.lut1bit.astype(int) ==
                      1 - 2 * ((np.arange(256)[:, np.newaxis] >>
                                np.arange(8)) & 1))
        assert np.all(mark4.payload.lut2bit1[0] == -o2h)
        assert np.all(mark4.payload.lut2bit1[0x55] == 1.)
        assert np.all(mark4.payload.lut2bit1[0xaa] == -1.)
        assert np.all(mark4.payload.lut2bit1[0xff] == o2h)
        assert np.all(mark4.payload.lut2bit2[0] == -o2h)
        assert np.all(mark4.payload.lut2bit2[0xcc] == -1.)
        assert np.all(mark4.payload.lut2bit2[0x33] == 1.)
        assert np.all(mark4.payload.lut2bit2[0xff] == o2h)
        assert np.all(mark4.payload.lut2bit3[0] == -o2h)
        assert np.all(mark4.payload.lut2bit3[0xf0] == -1.)
        assert np.all(mark4.payload.lut2bit3[0x0f] == 1.)
        assert np.all(mark4.payload.lut2bit3[0xff] == o2h)

    def test_payload_reorder(self):
        """Test that bit reordering is consistent with mark5access."""
        check = np.array([738811025863578102], dtype='<u8').view('u8')
        expected = np.array([118, 209, 53, 244, 148, 217, 64, 10])
        assert np.all(reorder64(check).view(np.uint8) == expected)
        assert np.all(reorder32(check.view('u4')).view(np.uint8) ==
                      expected)

    def test_payload(self, tmpdir):
        with open(SAMPLE_FILE, 'rb') as fh:
            fh.seek(0xa88)
            header = mark4.Mark4Header.fromfile(fh, ntrack=64, decade=2010)
            payload = mark4.Mark4Payload.fromfile(fh, header)
        assert payload.nbytes == (20000 - 160) * 64 // 8
        assert len(payload) == (20000 - 160) * 4
        assert payload.shape == ((20000 - 160) * 4, 8)
        assert payload.size == 634880
        assert payload.ndim == 2
        # Check sample shape validity.
        assert payload.sample_shape == (8,)
        assert payload.sample_shape.nchan == 8
        assert payload.dtype == np.float32
        assert np.all(payload[0].astype(int) ==
                      np.array([-1, +1, +1, -3, -3, -3, +1, -1]))
        assert np.all(payload[1].astype(int) ==
                      np.array([+1, +1, -3, +1, +1, -3, -1, -1]))

        with open(str(tmpdir.join('test.m4')), 'w+b') as s:
            payload.tofile(s)
            s.seek(0)
            payload2 = mark4.Mark4Payload.fromfile(s, header)
            assert payload2 == payload
            with pytest.raises(EOFError):
                # Too few bytes.
                s.seek(100)
                mark4.Mark4Payload.fromfile(s, header)
        payload3 = mark4.Mark4Payload.fromdata(payload.data, header)
        assert payload3 == payload
        payload4 = mark4.Mark4Payload(payload.words, nchan=8, bps=2, fanout=4)
        assert payload4 == payload
        with pytest.raises(ValueError):
            # Wrong number of channels.
            mark4.Mark4Payload.fromdata(np.empty((payload.shape[0], 2)),
                                        header)
        with pytest.raises(ValueError):
            # Too few data.
            mark4.Mark4Payload.fromdata(payload[:100], header)

        with pytest.raises(ValueError):
            # Wrong data type
            mark4.Mark4Payload.fromdata(np.zeros((5000, 8), np.complex64),
                                        header)
        with pytest.raises(ValueError):
            # Wrong encoded data type for implied number of tracks of 32.
            mark4.Mark4Payload(payload.words, nchan=4, bps=2, fanout=4)
        with pytest.raises(ValueError):
            # Not little-endian encoded data.
            mark4.Mark4Payload(payload.words.astype('>u8'), header)
        with pytest.raises(ValueError):
            # Wrong number of tracks in encoded data.
            mark4.Mark4Payload(payload.words.view('<u4'), header)

    @pytest.mark.parametrize('item', (2, (), -1, slice(1, 3), slice(2, 4),
                                      slice(2, 4), slice(-3, None),
                                      (2, slice(3, 5)), (10, 4),
                                      (slice(None), 5)))
    def test_payload_getitem_setitem(self, item):
        with open(SAMPLE_FILE, 'rb') as fh:
            fh.seek(0xa88)
            header = mark4.Mark4Header.fromfile(fh, ntrack=64, decade=2010)
            payload = mark4.Mark4Payload.fromfile(fh, header)
        sel_data = payload.data[item]
        assert np.all(payload[item] == sel_data)
        payload2 = mark4.Mark4Payload(payload.words.copy(), header)
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

    def test_binary_file_reader(self):
        with mark4.open(SAMPLE_FILE, 'rb', decade=2010, ntrack=64) as fh:
            fh.locate_frame()
            assert fh.tell() == 0xa88
            header = mark4.Mark4Header.fromfile(fh, decade=2010, ntrack=64)
            fh.seek(0xa88)
            header2 = fh.read_header()
            current_pos = fh.tell()
            assert header2 == header
            frame_rate = fh.get_frame_rate()
            assert abs(frame_rate -
                       32 * u.MHz / header.samples_per_frame) < 1 * u.nHz
            assert fh.tell() == current_pos
            repr_fh = repr(fh)

        assert repr_fh.startswith('Mark4FileReader')
        assert 'ntrack=64, decade=2010, ref_time=None' in repr_fh

    def test_frame(self, tmpdir):
        with mark4.open(SAMPLE_FILE, 'rb', decade=2010, ntrack=64) as fh:
            fh.seek(0xa88)
            header = mark4.Mark4Header.fromfile(fh, ntrack=64, decade=2010)
            payload = mark4.Mark4Payload.fromfile(fh, header)
            fh.seek(0xa88)
            frame = fh.read_frame()

        assert frame.header == header
        assert frame.payload == payload
        assert frame.valid is True
        assert len(frame) == len(payload) + 640
        assert frame.sample_shape == payload.sample_shape
        assert frame.shape == (len(frame),) + frame.sample_shape
        assert frame.size == len(frame) * np.prod(frame.sample_shape)
        assert frame.ndim == payload.ndim
        assert frame == mark4.Mark4Frame(header, payload)
        data = frame.data
        assert np.all(data[:640] == 0.)
        assert np.all(data[640].astype(int) ==
                      np.array([-1, +1, +1, -3, -3, -3, +1, -1]))
        assert np.all(data[641].astype(int) ==
                      np.array([+1, +1, -3, +1, +1, -3, -1, -1]))
        # Check writing and round-trip.
        with open(str(tmpdir.join('test.m4')), 'w+b') as s:
            frame.tofile(s)
            s.seek(0)
            frame2 = mark4.Mark4Frame.fromfile(s, ntrack=64, decade=2010)
        assert frame2 == frame
        # Need frame.data here, since payload.data does not contain pieces
        # overwritten by header.
        frame3 = mark4.Mark4Frame.fromdata(frame.data, header)
        assert frame3 == frame
        frame4 = mark4.Mark4Frame.fromdata(frame.data, ntrack=64,
                                           decade=2010, **header)
        assert frame4 == frame
        header5 = header.copy()
        frame5 = mark4.Mark4Frame(header5, payload, valid=False)
        assert frame5.valid is False
        assert np.all(frame5.data == 0.)
        frame5.valid = True
        assert frame5 == frame
        frame5.valid = False
        assert np.all(frame5.data == 0.)

        # Check passing in a reference time.
        with mark4.open(SAMPLE_FILE, 'rb', ref_time=Time('2009:345:15:00:00'),
                        ntrack=64) as fh:
            fh.seek(0xa88)
            frame6 = fh.read_frame()
            assert frame6 == frame

        with mark4.open(SAMPLE_FILE, 'rb', ref_time=Time('2019:1:9:00:00'),
                        ntrack=64) as fh:
            fh.seek(0xa88)
            frame7 = fh.read_frame()
            assert frame7 == frame

    def test_frame_getitem_setitem(self):
        with mark4.open(SAMPLE_FILE, 'rb', ref_time=Time('2009:345:15:00:00'),
                        ntrack=64) as fh:
            fh.seek(0xa88)
            frame = fh.read_frame()
            header = frame.header
            data = frame.data
        # Check __getitem__.
        assert np.all(frame['magnitude_bit'] == header['magnitude_bit'])
        # Only invalid values (compare with data to ensure shape is right).
        assert np.all(frame[10:90] == data[10:90])
        assert np.all(frame[10:90:5] == data[10:90:5])
        assert np.all(frame[10:90:5, :4] == data[10:90:5, :4])
        # Mixed invalid, valid.
        assert np.all(frame[635:655] == data[635:655])
        # Check we do stepping correctly around the boundary.
        for start in range(634, 642):
            assert np.all(frame[start:655:5] == data[start:655:5])
        assert np.all(frame[635:655:5, 5] == data[635:655:5, 5])
        # All valid.
        assert np.all(frame[935:955] == data[935:955])
        assert np.all(frame[935:955:5] == data[935:955:5])
        assert np.all(frame[935:955:5, 5] == data[935:955:5, 5])
        # Check scalar, including whether the shape is correct.
        assert frame[935].shape == data[935].shape
        assert np.all(frame[639] == data[639])
        assert np.all(frame[640] == data[640])
        # Check __setitem__.
        # Default frame read from a file is not mutable.
        with pytest.raises(ValueError):
            frame[640:] = 0.
        # So create a mutable version of the frame.
        frame = mark4.Mark4Frame.fromdata(frame.data, header.copy())
        # Check that setting invalid part has no effect.
        frame[10:90] == 1.
        frame[10:90:5] == 1.
        frame[10:90:5, :4] == 1.
        assert np.all(frame[:640] == 0.)
        # Mixed invalid, valid.
        frame[635:655] = 1.
        assert np.all(frame[635:640] == 0.)
        assert np.all(frame[640:655] == 1.)
        frame[635:655] = data[635:655]
        assert np.all(frame[:] == data)
        # Check we do stepping correctly around the boundary.
        for start in range(634, 642):
            frame[start:655:5] = 1.
            valid_start = 640 + start % 5
            if start < 640:
                assert np.all(frame[start:640:5] == 0.)
            assert np.all(frame[valid_start:655:5] == 1.)
        frame[634:655] = data[634:655]
        frame[635:655:5, 5] = -data[635:655:5, 5]
        assert np.all(frame[635:655:5, 5] == -data[635:655:5, 5])
        assert np.all(frame[635:655:5, :5] == data[635:655:5, :5])
        assert np.all(frame[635:655:5, 6:] == data[635:655:5, 6:])
        # All valid.
        frame[935:955] = 1.
        assert np.all(frame[935:955] == 1.)
        frame[935:955:5] = -1.
        assert np.all(frame[935:955:5] == -1.)
        assert np.all(frame[936:955:5] == 1.)
        frame[935:955:5, 5] = 1.
        assert np.all(frame[935:955:5, 5] == 1.)
        assert np.all(frame[935:955:5, :5] == -1.)
        frame[935:955] = data[935:955]
        # Check scalar, including whether the shape is correct.
        frame[935] = -data[935]
        assert np.all(frame[935] == -data[935])
        assert np.all(frame[934] == data[934])
        assert np.all(frame[936] == data[936])
        frame[:] = data
        assert np.all(frame[:] == data)
        # Finally, check that __getitem__ always returns 0 for invalid data.
        frame.valid = False
        assert frame[655, 0] == 0.
        assert np.all(frame[930:950] == 0.)
        assert np.all(frame[630:650:5, :4] == 0.)
        frame.valid = True
        # And check propagation of strings to header.
        frame['bcd_headstack2'] = 0
        assert np.all(frame.header['bcd_headstack2'] == 0.)

    def test_header_times(self):
        with mark4.open(SAMPLE_FILE, 'rb', decade=2010, ntrack=64) as fh:
            fh.seek(0xa88)
            header0 = mark4.Mark4Header.fromfile(fh, ntrack=64, decade=2010)
            start_time = header0.time
            # Use frame size, since header adds to payload.
            samples_per_frame = header0.frame_nbytes * 8 // 2 // 8
            frame_rate = 32. * u.MHz / samples_per_frame
            frame_duration = 1. / frame_rate
            fh.seek(0xa88)
            for frame_nr in range(100):
                try:
                    frame = fh.read_frame()
                except EOFError:
                    break
                header_time = frame.header.time
                expected = start_time + frame_nr * frame_duration
                assert abs(header_time - expected) < 1. * u.ns

    def test_find_header(self, tmpdir):
        # Below, the tests set the file pointer to very close to a header,
        # since otherwise they run *very* slow.  This is somehow related to
        # pytest, since speed is not a big issue running stuff on its own.
        with mark4.open(SAMPLE_FILE, 'rb', decade=2010) as fh:
            fh.seek(0xa88)
            header0 = mark4.Mark4Header.fromfile(fh, ntrack=64, decade=2010)
            fh.seek(0)
            header_0 = fh.find_header()
            assert fh.tell() == 0xa88
            assert fh.ntrack == 64
            assert header_0 == header0
            fh.seek(0xa89)
            header_0xa89 = fh.find_header()
            assert fh.tell() == 0xa88 + header0.frame_nbytes
            fh.seek(160000)
            header_160000f = fh.find_header(forward=True)
            assert fh.tell() == 0xa88 + header0.frame_nbytes
            fh.seek(0xa87)
            header_0xa87b = fh.find_header(forward=False)
            assert header_0xa87b is None
            assert fh.tell() == 0xa87
            fh.seek(0xa88)
            header_0xa88f = fh.find_header()
            assert fh.tell() == 0xa88
            fh.seek(0xa88)
            header_0xa88b = fh.find_header(forward=False)
            assert fh.tell() == 0xa88
            fh.seek(0xa88 + 100)
            header_100b = fh.find_header(forward=False)
            assert fh.tell() == 0xa88
            fh.seek(-10000, 2)
            header_m10000b = fh.find_header(forward=False)
            assert fh.tell() == 0xa88 + 2*header0.frame_nbytes
            fh.seek(-300, 2)
            header_end = fh.find_header(forward=True)
            assert header_end is None
        assert header_100b == header_0
        assert header_0xa88f == header_0
        assert header_0xa88b == header_0
        assert header_0xa89 == header_160000f
        assert abs(header_160000f.time - header_0.time - 2.5*u.ms) < 1.*u.ns
        assert abs(header_m10000b.time - header_0.time - 5*u.ms) < 1.*u.ns
        # test small file
        with open(SAMPLE_FILE, 'rb') as fh:
            # One that simply is too small altogether
            m4_test = str(tmpdir.join('test.m4'))
            with open(m4_test, 'w+b') as s, mark4.open(s, 'rb',
                                                       ntrack=64) as fh_short:
                s.write(fh.read(80000))
                fh_short.seek(100)
                assert fh_short.find_header() is None
                assert fh_short.tell() == 100
                assert fh_short.find_header(forward=False) is None
                assert fh_short.tell() == 100

            # And one that could fit one frame, but doesn't.
            with open(m4_test, 'w+b') as s, mark4.open(s, 'rb',
                                                       ntrack=64) as fh_short:
                fh.seek(0)
                s.write(fh.read(162690))
                fh_short.seek(200)
                assert fh_short.find_header() is None
                assert fh_short.tell() == 200
                assert fh_short.find_header(forward=False) is None
                assert fh_short.tell() == 200

            # now add enough that the file does include a complete header.
            with open(m4_test, 'w+b') as s, mark4.open(s, 'rb',
                                                       ntrack=64) as fh_short2:
                fh.seek(0)
                s.write(fh.read(163000))
                s.seek(0)
                fh_short2.seek(100)
                header_100f = fh_short2.find_header()
                assert fh_short2.tell() == 0xa88
                fh_short2.seek(-1000, 2)
                header_m1000b = fh_short2.find_header(forward=False)
                assert fh_short2.tell() == 0xa88
            assert header_100f == header0
            assert header_m1000b == header0

    def test_determine_ntrack(self):
        with mark4.open(SAMPLE_FILE, 'rb', ntrack=64) as fh:
            offset0 = fh.locate_frame()
            assert offset0 == 2696
        with mark4.open(SAMPLE_FILE, 'rb') as fh:
            assert fh.ntrack is None
            ntrack = fh.determine_ntrack()
            assert ntrack == fh.ntrack == 64
            assert fh.fh_raw.tell() == offset0

        with mark4.open(SAMPLE_32TRACK, 'rb', ntrack=32) as fh:
            # Seek past first frame header; find second frame.
            fh.seek(10000)
            offset0 = fh.locate_frame()
            assert offset0 == 89656
        with mark4.open(SAMPLE_32TRACK, 'rb') as fh:
            fh.seek(10000)
            ntrack = fh.determine_ntrack()
            assert fh.fh_raw.tell() == offset0
            assert ntrack == fh.ntrack == 32

        with mark4.open(SAMPLE_32TRACK_FANOUT2, 'rb', ntrack=32) as fh:
            offset0 = fh.locate_frame()
            assert offset0 == 17436
        with mark4.open(SAMPLE_32TRACK_FANOUT2, 'rb') as fh:
            ntrack = fh.determine_ntrack()
            assert fh.fh_raw.tell() == offset0
            assert ntrack == fh.ntrack == 32
            assert fh.ntrack == 32

    def test_filestreamer(self, tmpdir):
        with mark4.open(SAMPLE_FILE, 'rb') as fh:
            fh.seek(0xa88)
            header = mark4.Mark4Header.fromfile(fh, ntrack=64, decade=2010)

        with mark4.open(SAMPLE_FILE, 'rs', sample_rate=32*u.MHz,
                        ntrack=64, decade=2010) as fh:
            assert header == fh.header0
            assert fh.samples_per_frame == 80000
            assert fh.sample_shape == (8,)
            assert fh.shape == (2 * fh.samples_per_frame,) + fh.sample_shape
            assert fh.size == np.prod(fh.shape)
            assert fh.ndim == len(fh.shape)
            assert fh.sample_rate == 32 * u.MHz
            record = fh.read(642)
            assert fh.tell() == 642
            # regression test against #4, of incorrect frame offsets.
            fh.seek(80000 + 639)
            record2 = fh.read(2)
            assert fh.tell() == 80641
            # Raw file should be just after frame 1.
            assert fh.fh_raw.tell() == 0xa88 + 2 * fh.header0.frame_nbytes
            # Test seeker works with both int and str values for whence
            assert fh.seek(13, 0) == fh.seek(13, 'start')
            assert fh.seek(-13, 2) == fh.seek(-13, 'end')
            fhseek_int = fh.seek(17, 1)
            fh.seek(-17, 'current')
            fhseek_str = fh.seek(17, 'current')
            assert fhseek_int == fhseek_str
            with pytest.raises(ValueError):
                fh.seek(0, 'last')
            fh.seek(1, 'end')
            with pytest.raises(EOFError):
                fh.read()

        assert record.shape == (642, 8)
        assert np.all(record[:640] == 0.)
        assert np.all(record.astype(int)[640] ==
                      np.array([-1, +1, +1, -3, -3, -3, +1, -1]))
        assert np.all(record.astype(int)[641] ==
                      np.array([+1, +1, -3, +1, +1, -3, -1, -1]))
        assert record2.shape == (2, 8)
        assert np.all(record2[0] == 0.)
        assert not np.any(record2[1] == 0.)

        # Check passing a time object into decade.
        with mark4.open(SAMPLE_FILE, 'rs', ntrack=64,
                        ref_time=Time('2018:364:23:59:59')) as fh:
            assert header == fh.header0
        with mark4.open(SAMPLE_FILE, 'rs', ntrack=64,
                        ref_time=Time(56039.5, format='mjd')) as fh:
            assert header == fh.header0

        # Test if _get_frame_rate automatic frame rate calculator works,
        # returns same header and payload info.
        with mark4.open(SAMPLE_FILE, 'rs', ntrack=64, decade=2010) as fh:
            assert header == fh.header0
            assert fh.samples_per_frame == 80000
            assert fh.sample_rate == 32 * u.MHz
            record3 = fh.read(642)

        assert np.all(record3 == record)

        # Test if automatic ntrack and frame rate detectors work together.
        with mark4.open(SAMPLE_FILE, 'rs', decade=2010) as fh:
            assert header == fh.header0
            assert fh.sample_rate == 32 * u.MHz
            fh.seek(80000 + 639)
            record4 = fh.read(2)

        assert np.all(record4 == record2)

        with mark4.open(SAMPLE_FILE, 'rs', sample_rate=32*u.MHz,
                        ntrack=64, decade=2010) as fh:
            start_time = fh.time
            record = fh.read()
            fh_raw_tell1 = fh.fh_raw.tell()
            stop_time = fh.time

        rewritten_file = str(tmpdir.join('rewritten.m4'))
        with mark4.open(rewritten_file, 'ws', sample_rate=32*u.MHz,
                        time=start_time, ntrack=64, bps=2, fanout=4) as fw:
            assert fw.sample_rate == 32 * u.MHz
            # write in bits and pieces and with some invalid data as well.
            fw.write(record[:11])
            fw.write(record[11:80000])
            fw.write(record[80000:], valid=False)
            assert fw.tell(unit='time') == stop_time

        with mark4.open(rewritten_file, 'rs', sample_rate=32*u.MHz,
                        ntrack=64, decade=2010, subset=[3, 4]) as fh:
            assert fh.time == start_time
            assert fh.time == fh.tell(unit='time')
            assert fh.sample_rate == 32 * u.MHz
            record5 = fh.read(160000)
            assert fh.time == stop_time
            assert fh.sample_shape == (2,)
            assert np.all(record5[:80000] == record[:80000, 3:5])
            assert np.all(record5[80000:] == 0.)

        # Check files can be made byte-for-byte identical.  Here, we use the
        # original header so we set stuff like head_stack, etc.
        with open(str(tmpdir.join('test.m4')), 'w+b') as s, \
                mark4.open(s, 'ws', header0=header,
                           sample_rate=32*u.MHz) as fw:
            fw.write(record)
            number_of_bytes = s.tell()
            assert number_of_bytes == fh_raw_tell1 - 0xa88

            s.seek(0)
            with open(SAMPLE_FILE, 'rb') as fr:
                fr.seek(0xa88)
                orig_bytes = fr.read(number_of_bytes)
                conv_bytes = s.read()
                assert conv_bytes == orig_bytes

        # Test that squeeze attribute works on read (including in-place read).
        with mark4.open(SAMPLE_FILE, 'rs', ntrack=64, decade=2010,
                        subset=0) as fh:
            assert fh.sample_shape == ()
            assert fh.read(1).shape == (1,)
            fh.seek(0)
            out = np.zeros(12)
            fh.read(out=out)
            assert fh.tell() == 12
            assert np.all(out == record[:12, 0])

        with mark4.open(SAMPLE_FILE, 'rs', ntrack=64, decade=2010,
                        subset=[0], squeeze=False) as fh:
            assert fh.subset == ([0],)
            assert fh.sample_shape == (1,)
            assert fh.sample_shape.nchan == 1
            assert fh.read(1).shape == (1, 1)
            fh.seek(0)
            out = np.zeros((12, 1))
            fh.read(out=out)
            assert fh.tell() == 12
            assert np.all(out.squeeze() == record[:12, 0])

        # Test writing across decades.
        start_time = Time('2019:365:23:59:59.9975', precision=9)
        decadal_file = str(tmpdir.join('decade.m4'))
        with mark4.open(decadal_file, 'ws', sample_rate=32*u.MHz,
                        time=start_time, ntrack=64, bps=2, fanout=4) as fw:
            # write in bits and pieces and with some invalid data as well.
            fw.write(record)

        with mark4.open(decadal_file, 'rs', sample_rate=32*u.MHz,
                        ntrack=64, decade=2010) as fh:
            assert abs(fh.start_time - start_time) < 1. * u.ns
            assert fh.header0.decade == 2010
            record6 = fh.read()
            assert np.all(record6 == record)
            assert (abs(fh.time - Time('2020:1:00:00:00.0025', precision=9)) <
                    1. * u.ns)
            assert fh._frame.header.decade == 2020

    # Test that writing an incomplete stream is possible, and that frame set is
    # appropriately marked as invalid.
    @pytest.mark.parametrize('fill_value', (0., -999.))
    def test_incomplete_stream(self, tmpdir, fill_value):
        m4_incomplete = str(tmpdir.join('incomplete.m4'))
        with catch_warnings(UserWarning) as w:
            with mark4.open(SAMPLE_FILE, 'rs', ntrack=64, decade=2010) as fr:
                record = fr.read(10)
                with mark4.open(m4_incomplete, 'ws', header0=fr.header0,
                                sample_rate=32*u.MHz,
                                ntrack=64, decade=2010) as fw:
                    fw.write(record)
        assert len(w) == 1
        assert 'partial buffer' in str(w[0].message)
        with mark4.open(m4_incomplete, 'rs', sample_rate=32*u.MHz,
                        ntrack=64, decade=2010, fill_value=fill_value) as fwr:
            assert np.all(fwr.read() == fill_value)
            assert fwr.fill_value == fill_value

    def test_corrupt_stream(self, tmpdir):
        with mark4.open(SAMPLE_FILE, 'rb', decade=2010, ntrack=64) as fh, \
                open(str(tmpdir.join('test.m4')), 'w+b') as s:
            fh.seek(0xa88)
            frame = fh.read_frame()
            # Write single frame to file.
            frame.tofile(s)
            # Now add lots of data without headers.
            for i in range(5):
                frame.payload.tofile(s)
            s.seek(0)
            # With too many payload samples for one frame, f2.locate_frame
            # will fail.
            with pytest.raises(AssertionError):
                f2 = mark4.open(s, 'rs', sample_rate=32*u.MHz,
                                ntrack=64, decade=2010)

        with mark4.open(SAMPLE_FILE, 'rb', decade=2010, ntrack=64) as fh, \
                open(str(tmpdir.join('test.m4')), 'w+b') as s:
            fh.seek(0xa88)
            frame0 = fh.read_frame()
            frame1 = fh.read_frame()
            frame0.tofile(s)
            frame1.tofile(s)
            # now add lots of data without headers.
            for i in range(15):
                frame1.payload.tofile(s)
            s.seek(0)
            with mark4.open(s, 'rs', sample_rate=32*u.MHz,
                            ntrack=64, decade=2010) as f2:
                assert f2.header0 == frame0.header
                with pytest.raises(ValueError):
                    f2._last_header

    def test_stream_invalid(self):
        with pytest.raises(ValueError):
            mark4.open('ts.dat', 's')

    def test_stream_missing_decade(self):
        with pytest.raises(TypeError):
            mark4.open(SAMPLE_FILE, 'rs', ntrack=64)


class Test32TrackFanout4():
    def test_locate_frame(self):
        with mark4.open(SAMPLE_32TRACK, 'rb') as fh:
            assert fh.locate_frame() == 9656
            assert fh.ntrack == 32

    def test_header(self):
        with open(SAMPLE_32TRACK, 'rb') as fh:
            fh.seek(9656)
            header = mark4.Mark4Header.fromfile(fh, ntrack=32, decade=2010)

        # Try initialising with properties instead of keywords.
        # Here, we let
        # * time imply the decade, bcd_unit_year, bcd_day, bcd_hour,
        #   bcd_minute, bcd_second, bcd_fraction;
        # * ntrack, samples_per_frame, bps define headstack_id, bcd_track_id,
        #   fan_out, and magnitude_bit;
        # * nsb defines lsb_output and converter_id.
        header1 = mark4.Mark4Header.fromvalues(
            ntrack=32, samples_per_frame=80000, bps=2, nsb=2, time=header.time,
            system_id=108)
        assert header1 == header

    def test_file_streamer(self, tmpdir):
        with mark4.open(SAMPLE_32TRACK, 'rs', sample_rate=32*u.MHz,
                        ntrack=32, decade=2010) as fh:
            header0 = fh.header0
            assert fh.samples_per_frame == 80000
            assert fh.sample_rate == 32 * u.MHz
            start_time = fh.start_time
            assert start_time.yday == '2015:011:01:23:10.48500'
            record = fh.read(160000)
            fh_raw_tell1 = fh.fh_raw.tell()
            assert fh_raw_tell1 == 169656

        assert np.all(record[:640] == 0.)
        # Data retrieved using: m5d ar/rg10a_ar_no0014 MKIV1_4-256-4-2 700
        assert np.all(record[640:644].astype(int) == np.array(
            [[-1, 3, -1, -3],
             [3, 3, -3, 1],
             [-3, -1, 1, -1],
             [1, 3, 1, 3]]))

        fl = str(tmpdir.join('test.m4'))
        with mark4.open(fl, 'ws', header0=header0, sample_rate=32*u.MHz) as fw:
            fw.write(record)
            number_of_bytes = fw.fh_raw.tell()
            assert number_of_bytes == fh_raw_tell1 - 9656

        # Note: this test would not work if we wrote only a single record.
        with mark4.open(fl, 'rs', sample_rate=32*u.MHz,
                        ntrack=32, decade=2010) as fh:
            assert fh.start_time == start_time
            record2 = fh.read(1000)
            assert np.all(record2 == record[:1000])

        with open(fl, 'rb') as fh, open(SAMPLE_32TRACK, 'rb') as fr:
            fr.seek(9656)
            orig_bytes = fr.read(number_of_bytes)
            conv_bytes = fh.read()
            assert conv_bytes == orig_bytes


class Test32TrackFanout2():
    def test_locate_frame(self):
        with mark4.open(SAMPLE_32TRACK_FANOUT2, 'rb') as fh:
            assert fh.locate_frame() == 17436
            assert fh.ntrack == 32

    def test_header(self):
        with open(SAMPLE_32TRACK_FANOUT2, 'rb') as fh:
            fh.seek(17436)
            header = mark4.Mark4Header.fromfile(fh, ntrack=32, decade=2010)

        # Try initialising with properties instead of keywords.
        # * time imply the decade, bcd_unit_year, bcd_day, bcd_hour,
        #   bcd_minute, bcd_second, bcd_fraction;
        # * ntrack, samples_per_frame, bps define headstack_id, bcd_track_id,
        #   fan_out, and magnitude_bit;
        # * header.converter since lsb_output and converter_id are somewhat
        #   non-standard
        header1 = mark4.Mark4Header.fromvalues(
            ntrack=32, samples_per_frame=40000, bps=2, time=header.time,
            system_id=108, converters=header.converters)
        assert header1 == header

    def test_file_streamer(self, tmpdir):
        with mark4.open(SAMPLE_32TRACK_FANOUT2, 'rs', sample_rate=16*u.MHz,
                        ntrack=32, decade=2010) as fh:
            header0 = fh.header0
            assert fh.samples_per_frame == 40000
            assert fh.sample_rate == 16 * u.MHz
            start_time = fh.start_time
            assert start_time.yday == '2017:063:04:42:26.02500'
            record = fh.read(80000)
            fh_raw_tell1 = fh.fh_raw.tell()
            assert fh_raw_tell1 == 160000 + 17436

        assert np.all(record[:320] == 0.)
        # Compare with: m5d vlbi_b1133/gk049c_ar_no0011.m5a MKIV1_2-128-8-2 700
        assert np.all(record[320:324].astype(int) == np.array(
            [[-1, -1, 3, 1, 3, 3, 1, 1],
             [-3, -3, 1, -1, -1, 3, -3, -1],
             [-1, -1, -3, -1, 1, 1, -1, 1],
             [-1, -3, -1, 1, -1, 1, -1, 1]]))

        fl = str(tmpdir.join('test.m4'))
        with mark4.open(fl, 'ws', header0=header0, sample_rate=16*u.MHz) as fw:
            fw.write(record)
            number_of_bytes = fw.fh_raw.tell()
            assert number_of_bytes == fh_raw_tell1 - 17436

        # Note: this test would not work if we wrote only a single record.
        with mark4.open(fl, 'rs', ntrack=32, decade=2010,
                        sample_rate=16*u.MHz) as fh:
            assert fh.start_time == start_time
            record2 = fh.read(1000)
            assert np.all(record2 == record[:1000])

        with open(fl, 'rb') as fh, open(SAMPLE_32TRACK_FANOUT2, 'rb') as fr:
            fr.seek(17436)
            orig_bytes = fr.read(number_of_bytes)
            conv_bytes = fh.read()
            assert conv_bytes == orig_bytes


class Test16TrackFanout4():
    def test_locate_frame(self):
        with mark4.open(SAMPLE_16TRACK, 'rb') as fh:
            assert fh.locate_frame() == 22124
            assert fh.ntrack == 16

    def test_header(self):
        with open(SAMPLE_16TRACK, 'rb') as fh:
            fh.seek(22124)
            header = mark4.Mark4Header.fromfile(fh, ntrack=16, decade=2010)

        # Try initialising with properties instead of keywords.
        # * time imply the decade, bcd_unit_year, bcd_day, bcd_hour,
        #   bcd_minute, bcd_second, bcd_fraction;
        # * ntrack, samples_per_frame, bps define headstack_id, bcd_track_id,
        #   fan_out, and magnitude_bit;
        # * nsb = 1 sets lsb_output and converter_id
        header1 = mark4.Mark4Header.fromvalues(
            ntrack=16, samples_per_frame=80000, bps=2, time=header.time,
            system_id=108, nsb=1)
        assert header1 == header

    def test_file_streamer(self, tmpdir):
        with mark4.open(SAMPLE_16TRACK, 'rs', sample_rate=32*u.MHz,
                        ntrack=16, decade=2010) as fh:
            header0 = fh.header0
            assert fh.samples_per_frame == 80000
            assert fh.sample_rate == 32 * u.MHz
            start_time = fh.start_time
            assert start_time.yday == '2013:307:06:00:00.77000'
            record = fh.read(160000)
            fh_raw_tell1 = fh.fh_raw.tell()
            assert fh_raw_tell1 == 80000 + 22124
            fh.fh_raw.seek(0)
            preheader_junk = fh.fh_raw.read(22124)

        assert np.all(record[:640] == 0.)
        # Compare with m5d ar/gs033a_ar_no0055.m5a MKIV1_4-128-2-2 1000, and
        # taking first 28 payload samples:
        m5access_data = np.array(
            [[3, -3, -1, 1, 1, 1, 1, -1, -3, 3, 3, -1, -1, 3,
              -1, -1, 3, -3, 1, -3, -3, -1, 3, -3, -3, -3, 3, 1],
             [1, 1, -3, -3, 3, 1, -1, 1, 3, 1, 1, 3, -3, -1,
              -1, 1, 1, -3, -1, -1, -3, -3, 1, 3, 1, -1, 1, 3]])
        assert np.all(record[640:668].astype(int) == m5access_data.T)

        fl = str(tmpdir.join('test.m4'))
        with mark4.open(fl, 'ws', header0=header0, sample_rate=32*u.MHz) as fw:
            fw.fh_raw.write(preheader_junk)
            fw.write(record)
            number_of_bytes = fw.fh_raw.tell()
            assert number_of_bytes == fh_raw_tell1

        # Note: this test would not work if we wrote only a single record.
        with mark4.open(fl, 'rs', sample_rate=32*u.MHz,
                        ntrack=16, decade=2010) as fh:
            assert fh.start_time == start_time
            record2 = fh.read()
            assert np.all(record2 == record)

        with open(fl, 'rb') as fh, open(SAMPLE_16TRACK, 'rb') as fr:
            orig_bytes = fr.read(number_of_bytes)
            conv_bytes = fh.read()
            assert conv_bytes == orig_bytes
