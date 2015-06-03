import io
import numpy as np
from astropy import units as u
from astropy.tests.helper import pytest
from astropy.time import Time
from .. import mark4


class TestMark4(object):
    def test_header_stream(self):
        with open('sample.m4', 'rb') as fh:
            fh.seek(0xa88)
            stream = np.fromfile(fh, dtype=np.uint64, count=5 * 32)
        # check sync words in right place
        assert np.all(stream[64:80] == 0xffffffffffffffff)
        assert mark4.header.crc12.check(stream)
        assert np.all(mark4.header.crc12(stream[:-12]) == stream[-12:])
        words = mark4.header.stream2words(stream)
        assert np.all(mark4.header.words2stream(words) == stream)

    def test_header(self):
        with open('sample.m4', 'rb') as fh:
            fh.seek(0xa88)
            header = mark4.Mark4Header.fromfile(fh, ntrack=64, decade=2010)
        assert header.size == 160 * 64 // 8
        assert header.fanout == 4
        assert header.bps == 2
        assert header.nchan == 8
        assert int(header.time.mjd) == 56824
        assert header.time.isot == '2014-06-16T07:38:12.47500'
        assert header.framesize == 20000 * 64 // 8
        assert header.payloadsize == header.framesize - header.size
        with io.BytesIO() as s:
            header.tofile(s)
            s.seek(0)
            header2 = mark4.Mark4Header.fromfile(s, header.ntrack,
                                                 header.decade)
        assert header2 == header
        header3 = mark4.Mark4Header.fromkeys(header.ntrack, header.decade,
                                             **header)
        assert header3 == header
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
        # Check decade.
        header5 = mark4.Mark4Header(header.words, decade=2015)
        assert header5.time == header.time
        header6 = mark4.Mark4Header(header.words, decade=2019)
        assert header6.time == header.time
