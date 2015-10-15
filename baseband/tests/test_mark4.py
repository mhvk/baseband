import io
import os
import numpy as np
from astropy import units as u
from astropy.tests.helper import pytest
from .. import mark4


SAMPLE_FILE = os.path.join(os.path.dirname(__file__), 'sample.m4')


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

    def test_payload(self):
        with open(SAMPLE_FILE, 'rb') as fh:
            fh.seek(0xa88)
            header = mark4.Mark4Header.fromfile(fh, ntrack=64, decade=2010)
            payload = mark4.Mark4Payload.fromfile(fh, header)
        assert payload.size == (20000 - 160) * 64 // 8
        assert payload.shape == ((20000 - 160) * 4, 8)
        assert payload.dtype == np.float32
        assert np.all(payload.data[0].astype(int) ==
                      np.array([-1, +1, +1, -3, -3, -3, +1, -1]))
        assert np.all(payload.data[1].astype(int) ==
                      np.array([+1, +1, -3, +1, +1, -3, -1, -1]))
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

    def test_frame(self):
        with mark4.open(SAMPLE_FILE, 'rb') as fh:
            fh.seek(0xa88)
            header = mark4.Mark4Header.fromfile(fh, ntrack=64, decade=2010)
            payload = mark4.Mark4Payload.fromfile(fh, header)
            fh.seek(0xa88)
            frame = fh.read_frame(ntrack=64, decade=2010)

        assert frame.header == header
        assert frame.payload == payload
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
        frame3 = mark4.Mark4Frame.fromdata(frame.data, frame.header)
        assert frame3 == frame

    def test_filestreamer(self):
        with mark4.open(SAMPLE_FILE, 'rb') as fh:
            fh.seek(0xa88)
            header = mark4.Mark4Header.fromfile(fh, ntrack=64, decade=2010)

        with mark4.open(SAMPLE_FILE, 'rs', ntrack=64, decade=2010,
                        sample_rate=32*u.MHz) as fh:
            assert header == fh.header0
            record = fh.read(642)
            assert fh.offset == 642

        assert record.shape == (642, 8)
        assert np.all(record[:640] == 0.)
        assert np.all(record.astype(int)[640] ==
                      np.array([-1, +1, +1, -3, -3, -3, +1, -1]))
        assert np.all(record.astype(int)[641] ==
                      np.array([+1, +1, -3, +1, +1, -3, -1, -1]))
