# Licensed under the GPLv3 - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import pytest
import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.tests.helper import catch_warnings
from ... import mark4
from ...vlbi_base.encoding import OPTIMAL_2BIT_HIGH
from ..header import Mark4TrackHeader
from ..payload import reorder32, reorder64
from ...data import (SAMPLE_MARK4 as SAMPLE_FILE,
                     SAMPLE_MARK4_32TRACK as SAMPLE_32TRACK,
                     SAMPLE_MARK4_32TRACK_FANOUT2 as SAMPLE_32TRACK_FANOUT2)

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
# Initially this seemed strange, since PAYLOADSIZE=20000 leads to 80000
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
        assert header.size == 160 * 64 // 8
        assert header.fanout == 4
        assert header.bps == 2
        assert not np.all(~header['magnitude_bit'])
        assert header.nchan == 8
        assert int(header.time.mjd) == 56824
        assert header.time.isot == '2014-06-16T07:38:12.47500'
        assert header.samples_per_frame == 20000 * 4
        assert header.framesize == 20000 * 64 // 8
        assert header.payloadsize == header.framesize - header.size
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
        # Check decade.
        header5 = mark4.Mark4Header(header.words, decade=2015)
        assert header5.time == header.time
        header6 = mark4.Mark4Header(header.words, decade=2019)
        assert header6.time == header.time
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
            header7.framesize = header.framesize
        with pytest.raises(AttributeError):
            header7.payloadsize = header.payloadsize
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
        with pytest.raises(AssertionError):  # missing decade
            Mark4TrackHeader.fromvalues(**header9)
        with pytest.raises(IndexError):
            header[65]
        with pytest.raises(ValueError):
            header[np.array([[0, 1], [2, 3]])]
        # check that one can construct crazy headers, even if not much works.
        header12 = mark4.Mark4Header(None, ntrack=53, decade=2010,
                                     verify=False)
        header12.time = header.time
        assert header12.ntrack == 53
        assert abs(header12.time - header.time) < 1. * u.ns

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
        assert payload.size == (20000 - 160) * 64 // 8
        assert payload.shape == ((20000 - 160) * 4, 8)
        # Check sample shape validity
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

    def test_frame(self, tmpdir):
        with mark4.open(SAMPLE_FILE, 'rb') as fh:
            fh.seek(0xa88)
            header = mark4.Mark4Header.fromfile(fh, ntrack=64, decade=2010)
            payload = mark4.Mark4Payload.fromfile(fh, header)
            fh.seek(0xa88)
            frame = fh.read_frame(ntrack=64, decade=2010)

        assert frame.header == header
        assert frame.payload == payload
        assert frame.valid is True
        assert frame == mark4.Mark4Frame(header, payload)
        assert np.all(frame.data[:640] == 0.)
        assert np.all(frame.data[640].astype(int) ==
                      np.array([-1, +1, +1, -3, -3, -3, +1, -1]))
        assert np.all(frame.data[641].astype(int) ==
                      np.array([+1, +1, -3, +1, +1, -3, -1, -1]))
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
        # check __getitem__
        assert np.all(frame['magnitude_bit'] == header['magnitude_bit'])
        # indexing with a slice should not (yet) work, since we cannot just
        # take the slice of the payload.
        with pytest.raises(IndexError):
            frame[10:20]

    def test_header_times(self):
        with mark4.open(SAMPLE_FILE, 'rb') as fh:
            fh.seek(0xa88)
            header0 = mark4.Mark4Header.fromfile(fh, ntrack=64, decade=2010)
            start_time = header0.time
            # use framesize, since header adds to payload.
            samples_per_frame = header0.framesize * 8 // 2 // 8
            frame_rate = 32. * u.MHz / samples_per_frame
            frame_duration = 1. / frame_rate
            fh.seek(0xa88)
            for frame_nr in range(100):
                try:
                    frame = fh.read_frame(ntrack=64, decade=2010)
                except EOFError:
                    break
                header_time = frame.header.time
                expected = start_time + frame_nr * frame_duration
                assert abs(header_time - expected) < 1. * u.ns

    def test_find_header(self, tmpdir):
        # Below, the tests set the file pointer to very close to a header,
        # since otherwise they run *very* slow.  This is somehow related to
        # pytest, since speed is not a big issue running stuff on its own.
        with mark4.open(SAMPLE_FILE, 'rb') as fh:
            fh.seek(0xa88)
            header0 = mark4.Mark4Header.fromfile(fh, ntrack=64, decade=2010)
            fh.seek(0)
            header_0 = fh.find_header(template_header=header0)
            assert fh.tell() == 0xa88
            fh.seek(0xa89)
            header_0xa89 = fh.find_header(template_header=header0)
            assert fh.tell() == 0xa88 + header0.framesize
            fh.seek(160000)
            header_160000f = fh.find_header(template_header=header0,
                                            forward=True)
            assert fh.tell() == 0xa88 + header0.framesize
            fh.seek(0xa87)
            header_0xa87b = fh.find_header(ntrack=header0.ntrack,
                                           decade=header0.decade,
                                           forward=False)
            assert header_0xa87b is None
            assert fh.tell() == 0xa87
            fh.seek(0xa88)
            header_0xa88f = fh.find_header(ntrack=header0.ntrack,
                                           decade=header0.decade)
            assert fh.tell() == 0xa88
            fh.seek(0xa88)
            header_0xa88b = fh.find_header(template_header=header0,
                                           forward=False)
            assert fh.tell() == 0xa88
            fh.seek(0xa88 + 100)
            header_100b = fh.find_header(template_header=header0,
                                         forward=False)
            assert fh.tell() == 0xa88
            fh.seek(-10000, 2)
            header_m10000b = fh.find_header(template_header=header0,
                                            forward=False)
            assert fh.tell() == 0xa88 + 2*header0.framesize
            fh.seek(-300, 2)
            header_end = fh.find_header(template_header=header0, forward=True)
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
            with open(m4_test, 'w+b') as s, mark4.open(s, 'rb') as fh_short:
                s.write(fh.read(80000))
                fh_short.seek(100)
                assert fh_short.find_header(template_header=header0) is None
                assert fh_short.tell() == 100
                assert fh_short.find_header(template_header=header0,
                                            forward=False) is None
                assert fh_short.tell() == 100

            # And one that could fit one frame, but doesn't.
            with open(m4_test, 'w+b') as s, mark4.open(s, 'rb') as fh_short:
                fh.seek(0)
                s.write(fh.read(162690))
                fh_short.seek(200)
                assert fh_short.find_header(template_header=header0) is None
                assert fh_short.tell() == 200
                assert fh_short.find_header(template_header=header0,
                                            forward=False) is None
                assert fh_short.tell() == 200

            # now add enough that the file does include a complete header.
            with open(m4_test, 'w+b') as s, mark4.open(s, 'rb') as fh_short2:
                fh.seek(0)
                s.write(fh.read(163000))
                s.seek(0)
                fh_short2.seek(100)
                header_100f = fh_short2.find_header(template_header=header0)
                assert fh_short2.tell() == 0xa88
                fh_short2.seek(-1000, 2)
                header_m1000b = fh_short2.find_header(template_header=header0,
                                                      forward=False)
                assert fh_short2.tell() == 0xa88
            assert header_100f == header0
            assert header_m1000b == header0

    def test_filestreamer(self, tmpdir):
        with mark4.open(SAMPLE_FILE, 'rb') as fh:
            fh.seek(0xa88)
            header = mark4.Mark4Header.fromfile(fh, ntrack=64, decade=2010)

        with mark4.open(SAMPLE_FILE, 'rs', ntrack=64, decade=2010,
                        sample_rate=32*u.MHz) as fh:
            assert header == fh.header0
            # Raw file should be just after frame 0.
            assert fh.fh_raw.tell() == 0xa88 + fh.header0.framesize
            assert fh.samples_per_frame == 80000
            assert fh.size == 2 * fh.samples_per_frame
            record = fh.read(642)
            assert fh.tell() == 642
            # regression test against #4, of incorrect frame offsets.
            fh.seek(80000 + 639)
            record2 = fh.read(2)
            assert fh.tell() == 80641
            # Raw file should be just after frame 1.
            assert fh.fh_raw.tell() == 0xa88 + 2 * fh.header0.framesize
            # Test seeker works with both int and str values for whence
            assert fh.seek(13, 0) == fh.seek(13, 'start')
            assert fh.seek(-13, 2) == fh.seek(-13, 'end')
            fhseek_int = fh.seek(17, 1)
            fh.seek(-17, 'current')
            fhseek_str = fh.seek(17, 'current')
            assert fhseek_int == fhseek_str
            with pytest.raises(ValueError):
                fh.seek(0, 'last')

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
                        decade=Time('2018:364:23:59:59')) as fh:
            assert header == fh.header0
        with mark4.open(SAMPLE_FILE, 'rs', ntrack=64,
                        decade=Time(56039.5, format='mjd')) as fh:
            assert header == fh.header0

        # Test if _get_frame_rate automatic frame rate calculator works,
        # returns same header and payload info.
        with mark4.open(SAMPLE_FILE, 'rs', ntrack=64, decade=2010) as fh:
            assert header == fh.header0
            assert fh.frames_per_second * fh.samples_per_frame == 32000000
            record3 = fh.read(642)

        assert np.all(record3 == record)

        with mark4.open(SAMPLE_FILE, 'rs', ntrack=64, decade=2010,
                        sample_rate=32*u.MHz) as fh:
            start_time = fh.current_time
            record = fh.read()
            fh_raw_tell1 = fh.fh_raw.tell()
            stop_time = fh.current_time

        rewritten_file = str(tmpdir.join('rewritten.m4'))
        with mark4.open(rewritten_file, 'ws', sample_rate=32*u.MHz,
                        time=start_time, ntrack=64, bps=2, fanout=4) as fw:
            # write in bits and pieces and with some invalid data as well.
            fw.write(record[:11])
            fw.write(record[11:80000])
            fw.write(record[80000:], invalid_data=True)
            assert fw.tell(unit='time') == stop_time

        with mark4.open(rewritten_file, 'rs', ntrack=64, decade=2010,
                        sample_rate=32*u.MHz, thread_ids=[3, 4]) as fh:
            assert fh.current_time == start_time
            assert fh.current_time == fh.tell(unit='time')
            record2 = fh.read(160000)
            assert fh.current_time == stop_time
            assert np.all(record2[:80000] == record[:80000, 3:5])
            assert np.all(record2[80000:] == 0.)

        # Check files can be made byte-for-byte identical.  Here, we use the
        # original header so we set stuff like head_stack, etc.
        with open(str(tmpdir.join('test.m4')), 'w+b') as s, \
                mark4.open(s, 'ws', header=header, sample_rate=32*u.MHz) as fw:
            fw.write(record)
            number_of_bytes = s.tell()
            assert number_of_bytes == fh_raw_tell1 - 0xa88

            s.seek(0)
            with open(SAMPLE_FILE, 'rb') as fr:
                fr.seek(0xa88)
                orig_bytes = fr.read(number_of_bytes)
                conv_bytes = s.read()
                assert conv_bytes == orig_bytes

        # Test that squeeze attribute works on read (including in-place read)
        # and write, but can be turned off if needed.
        with mark4.open(SAMPLE_FILE, 'rs', ntrack=64, decade=2010) as fh:
            assert fh.sample_shape == (8,)
            assert fh.sample_shape.nchan == 8
            assert fh.read(1).shape == (8,)
            fh.seek(0)
            out = np.zeros((12, 8))
            fh.read(out=out)
            assert fh.tell() == 12
            assert np.all(out == record[:12])

        with mark4.open(SAMPLE_FILE, 'rs', ntrack=64, decade=2010,
                        thread_ids=[0], squeeze=False) as fh:
            assert fh.sample_shape == (1,)
            assert fh.sample_shape.nchan == 1
            assert fh.read(1).shape == (1, 1)
            fh.seek(0)
            out = np.zeros((12, 1))
            fh.read(out=out)
            assert fh.tell() == 12
            assert np.all(out.squeeze() == record[:12, 0])

        with mark4.open(str(tmpdir.join('test.m4')), 'ws',
                        sample_rate=32*u.MHz, time=start_time,
                        ntrack=64, bps=1, fanout=4) as fw:
            assert fw.sample_shape == (16,)
            assert fw.sample_shape.nchan == 16

    # Test that writing an incomplete stream is possible, and that frame set is
    # appropriately marked as invalid.
    def test_incomplete_stream(self, tmpdir):
        m4_incomplete = str(tmpdir.join('incomplete.m4'))
        with catch_warnings(UserWarning) as w:
            with mark4.open(SAMPLE_FILE, 'rs', ntrack=64, decade=2010) as fr:
                record = fr.read(10)
                with mark4.open(m4_incomplete, 'ws', header=fr.header0,
                                ntrack=64, decade=2010,
                                sample_rate=32*u.MHz) as fw:
                    fw.write(record)
        assert len(w) == 1
        assert 'partial buffer' in str(w[0].message)
        with mark4.open(m4_incomplete, 'rs', ntrack=64, decade=2010,
                        frames_per_second=400) as fwr:
            assert not fwr._frame.valid
            assert np.all(fwr.read() ==
                          fwr._frame.invalid_data_value)

    def test_corrupt_stream(self, tmpdir):
        with mark4.open(SAMPLE_FILE, 'rb') as fh, \
                open(str(tmpdir.join('test.m4')), 'w+b') as s:
            fh.seek(0xa88)
            frame = fh.read_frame(ntrack=64, decade=2010)
            frame.tofile(s)
            # now add lots of data without headers.
            for i in range(15):
                frame.payload.tofile(s)
            s.seek(0)
            with mark4.open(s, 'rs', ntrack=64, decade=2010,
                            sample_rate=32*u.MHz) as f2:
                assert f2.header0 == frame.header
                with pytest.raises(ValueError):
                    f2._header_last

    def test_stream_invalid(self):
        with pytest.raises(ValueError):
            mark4.open('ts.dat', 's')


class Test32TrackFanout4():
    def test_find_frame(self):
        with mark4.open(SAMPLE_32TRACK, 'rb') as fh:
            assert fh.find_frame(ntrack=32) == 9656

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
        with mark4.open(SAMPLE_32TRACK, 'rs', ntrack=32, decade=2010,
                        frames_per_second=400) as fh:
            header0 = fh.header0
            assert fh.samples_per_frame == 80000
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
        with mark4.open(fl, 'ws', header=header0, frames_per_second=400) as fw:
            fw.write(record)
            number_of_bytes = fw.fh_raw.tell()
            assert number_of_bytes == fh_raw_tell1 - 9656

        # Note: this test would not work if we wrote only a single record.
        with mark4.open(fl, 'rs', ntrack=32, decade=2010,
                        frames_per_second=400) as fh:
            assert fh.start_time == start_time
            record2 = fh.read(1000)
            assert np.all(record2 == record[:1000])

        with open(fl, 'rb') as fh, open(SAMPLE_32TRACK, 'rb') as fr:
            fr.seek(9656)
            orig_bytes = fr.read(number_of_bytes)
            conv_bytes = fh.read()
            assert conv_bytes == orig_bytes


class Test32TrackFanout2():
    def test_find_frame(self):
        with mark4.open(SAMPLE_32TRACK_FANOUT2, 'rb') as fh:
            assert fh.find_frame(ntrack=32) == 17436

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
        with mark4.open(SAMPLE_32TRACK_FANOUT2, 'rs', ntrack=32, decade=2010,
                        frames_per_second=400) as fh:
            header0 = fh.header0
            assert fh.samples_per_frame == 40000
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
        with mark4.open(fl, 'ws', header=header0, frames_per_second=400) as fw:
            fw.write(record)
            number_of_bytes = fw.fh_raw.tell()
            assert number_of_bytes == fh_raw_tell1 - 17436

        # Note: this test would not work if we wrote only a single record.
        with mark4.open(fl, 'rs', ntrack=32, decade=2010,
                        frames_per_second=400) as fh:
            assert fh.start_time == start_time
            record2 = fh.read(1000)
            assert np.all(record2 == record[:1000])

        with open(fl, 'rb') as fh, open(SAMPLE_32TRACK_FANOUT2, 'rb') as fr:
            fr.seek(17436)
            orig_bytes = fr.read(number_of_bytes)
            conv_bytes = fh.read()
            assert conv_bytes == orig_bytes
