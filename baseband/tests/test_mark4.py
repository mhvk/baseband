import io
import os
import numpy as np
from astropy import units as u
from astropy.tests.helper import pytest
from .. import mark4, vlbi_base
from ..mark4.header import Mark4TrackHeader
from ..mark4.payload import reorder32, reorder64


SAMPLE_FILE = os.path.join(os.path.dirname(__file__), 'sample.m4')
"""Mark 4 sample.

Created from one of our EVN PSR B1957+20 observations using
dd if=gp052d_ar_no0021 of=sample.m4 bs=128000 count=3
"""


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
            stream = np.fromfile(fh, dtype=np.uint64, count=5 * 32)
        # check sync words in right place
        assert np.all(stream[64:80] == 0xffffffffffffffff)
        assert mark4.header.crc12.check(stream)
        assert np.all(mark4.header.crc12(stream[:-12]) == stream[-12:])
        words = mark4.header.stream2words(stream)
        assert np.all(mark4.header.words2stream(words) == stream)

    def test_header(self):
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
        assert header.framesize == 20000 * 64 // 8
        assert header.payloadsize == header.framesize - header.size
        assert header.mutable is False
        assert repr(header).startswith('<Mark4Header bcd_headstack1: [0')
        with io.BytesIO() as s:
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
        # and ntrack, fanout, bps define headstack_id, bcd_track_id,
        # fan_out, magnitude_bit, and converter_id.
        header4 = mark4.Mark4Header.fromvalues(
            ntrack=64, fanout=4, bps=2, time=header.time,
            bcd_headstack1=0x3344, bcd_headstack2=0x1122,
            lsb_output=True, system_id=108)
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
        with pytest.raises(Exception):
            header7.ntrack = 51
        with pytest.raises(AssertionError):
            header7.framesize = header.framesize - 1
        with pytest.raises(AssertionError):
            header7.payloadsize = header.payloadsize - 1
        with pytest.raises(Exception):
            header7.framesize = header.framesize * 2  # implied ntrack=128
        header7.nchan = 16
        assert header7.nchan == 16 and header7.bps == 1
        # OK, this is silly, but why not...
        header7.time = header.time + np.arange(64) * 125 * u.ms
        assert len(header7.time) == 64
        assert np.all(abs(header7.time - header.time -
                          np.arange(64) * 125 * u.ms) < 1.*u.ns)
        with pytest.raises(ValueError):
            header7.time = header.time + np.arange(64) * u.year
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
        header12 = mark4.Mark4Header(None, ntrack=53, decade=2010,
                                     verify=False)
        with pytest.raises(ValueError):
            header12.ntrack = 53

    def test_decoding(self):
        """Check that look-up levels are consistent with mark5access."""
        o2h = vlbi_base.payload.OPTIMAL_2BIT_HIGH
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
        check = np.array([738811025863578102], dtype=np.uint64)
        expected = np.array([118, 209, 53, 244, 148, 217, 64, 10], np.uint8)
        assert np.all(reorder64(check).view(np.uint8) == expected)
        assert np.all(reorder32(check.view(np.uint32)).view(np.uint8) ==
                      expected)

    def test_payload(self):
        with open(SAMPLE_FILE, 'rb') as fh:
            fh.seek(0xa88)
            header = mark4.Mark4Header.fromfile(fh, ntrack=64, decade=2010)
            payload = mark4.Mark4Payload.fromfile(fh, header)
        assert payload.size == (20000 - 160) * 64 // 8
        assert payload.shape == ((20000 - 160) * 4, 8)
        assert payload.dtype == np.float32
        data = payload.data
        assert np.all(data[0].astype(int) ==
                      np.array([-1, +1, +1, -3, -3, -3, +1, -1]))
        assert np.all(data[1].astype(int) ==
                      np.array([+1, +1, -3, +1, +1, -3, -1, -1]))
        in_place = np.zeros_like(data)
        payload.todata(data=in_place)
        assert in_place is not data
        assert np.all(in_place == data)

        with io.BytesIO() as s:
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
        with pytest.raises(ValueError):
            # Wrong number of channels.
            mark4.Mark4Payload.fromdata(np.empty((payload.shape[0], 2)),
                                        header)
        with pytest.raises(ValueError):
            # Too few data.
            mark4.Mark4Payload.fromdata(payload.data[:100], header)

        with pytest.raises(ValueError):
            # Wrong data type
            mark4.Mark4Payload.fromdata(np.zeros((5000, 8), np.complex64),
                                        header)

    def test_frame(self):
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
        with io.BytesIO() as s:
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

    def test_header_times(self):
        with mark4.open(SAMPLE_FILE, 'rb') as fh:
            fh.seek(0xa88)
            header0 = mark4.Mark4Header.fromfile(fh, ntrack=64, decade=2010)
            time0 = header0.time
            # use framesize, since header adds to payload.
            samples_per_frame = header0.framesize * 8 // 2 // 8
            frame_rate = 32. * u.MHz / samples_per_frame
            frame_duration = 1./frame_rate
            fh.seek(0xa88)
            for frame_nr in range(100):
                try:
                    frame = fh.read_frame(ntrack=64, decade=2010)
                except EOFError:
                    break
                header_time = frame.header.time
                expected = time0 + frame_nr * frame_duration
                assert abs(header_time - expected) < 1. * u.ns

    def test_filestreamer(self):
        with mark4.open(SAMPLE_FILE, 'rb') as fh:
            fh.seek(0xa88)
            header = mark4.Mark4Header.fromfile(fh, ntrack=64, decade=2010)

        with mark4.open(SAMPLE_FILE, 'rs', ntrack=64, decade=2010,
                        sample_rate=32*u.MHz) as fh:
            assert header == fh.header0
            # Raw file should be just after frame 0.
            assert fh.fh_raw.tell() == 0xa88 + fh.header0.framesize
            record = fh.read(642)
            assert fh.tell() == 642
            # regression test against #4, of incorrect frame offsets.
            fh.seek(80000 + 639)
            record2 = fh.read(2)
            assert fh.tell() == 80641
            # Raw file should be just after frame 1.
            assert fh.fh_raw.tell() == 0xa88 + 2 * fh.header0.framesize

        assert record.shape == (642, 8)
        assert np.all(record[:640] == 0.)
        assert np.all(record.astype(int)[640] ==
                      np.array([-1, +1, +1, -3, -3, -3, +1, -1]))
        assert np.all(record.astype(int)[641] ==
                      np.array([+1, +1, -3, +1, +1, -3, -1, -1]))
        assert record2.shape == (2, 8)
        assert np.all(record2[0] == 0.)
        assert not np.any(record2[1] == 0.)

        with mark4.open(SAMPLE_FILE, 'rs', ntrack=64, decade=2010,
                        sample_rate=32*u.MHz) as fh:
            time0 = fh.tell(unit='time')
            record = fh.read(160000)
            fh_raw_tell1 = fh.fh_raw.tell()
            time1 = fh.tell(unit='time')

        with io.BytesIO() as s, mark4.open(s, 'ws', sample_rate=32*u.MHz,
                                           time=time0, ntrack=64, bps=2,
                                           fanout=4) as fw:
            fw.write(record)
            assert fw.tell(unit='time') == time1
            fw.fh_raw.flush()

            s.seek(0)
            fh = mark4.open(s, 'rs', ntrack=64, decade=2010,
                            sample_rate=32*u.MHz)
            assert fh.tell(unit='time') == time0
            record2 = fh.read(160000)
            assert fh.tell(unit='time') == time1
            assert np.all(record2 == record)

        # Check files can be made byte-for-byte identical.  Here, we use the
        # original header so we set stuff like head_stack, etc.
        with io.BytesIO() as s, mark4.open(s, 'ws', header=header,
                                           sample_rate=32*u.MHz) as fw:

            fw.write(record)
            fw.fh_raw.flush()
            number_of_bytes = s.tell()
            assert number_of_bytes == fh_raw_tell1 - 0xa88

            s.seek(0)
            with open(SAMPLE_FILE, 'rb') as fr:
                fr.seek(0xa88)
                orig_bytes = fr.read(number_of_bytes)
                conv_bytes = s.read()
                assert conv_bytes == orig_bytes
