import numpy as np
from astropy.time import Time
from astropy.tests.helper import pytest
from .. import vdif
from .. import mark5b


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
    def test_header(self):
        with open('vlba.m5a', 'rb') as fh:
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

        header2 = vdif.VDIFHeader.frombytes(header.tobytes())
        assert header2 == header
        header3 = vdif.VDIFHeader.fromkeys(**header)
        assert header3 == header
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

    def test_payload(self):
        with open('vlba.m5a', 'rb') as fh:
            header = vdif.VDIFHeader.fromfile(fh)
            payload = vdif.VDIFPayload.fromfile(fh, header)
        assert np.all(payload.data[:12, 0].astype(int) ==
                      np.array([1, 1, 1, -3, 1, 1, -3, -3, -3, 3, 3, -1]))
        payload2 = vdif.VDIFPayload.frombytes(payload.tobytes(), header)
        assert payload2 == payload
        payload3 = vdif.VDIFPayload.fromdata(payload.data, header)
        assert payload3 == payload
        with pytest.raises(ValueError):
            # Too few bytes.
            vdif.VDIFPayload.frombytes(payload.tobytes()[:100], header)
        with pytest.raises(ValueError):
            # Wrong number of channels.
            vdif.VDIFPayload.frombytes(payload.data[:, :4], header)
        with pytest.raises(ValueError):
            # Too few data.
            vdif.VDIFPayload.frombytes(payload.data[:100], header)

    def test_frame(self):
        with vdif.open('vlba.m5a', 'rb') as fh:
            header = vdif.VDIFHeader.fromfile(fh)
            payload = vdif.VDIFPayload.fromfile(fh, header)
            fh.seek(0)
            frame = fh.read_frame()

        assert frame.header == header
        assert frame.payload == payload
        assert frame == vdif.VDIFFrame(header, payload)
        assert np.all(frame.data[:12, 0].astype(int) ==
                      np.array([1, 1, 1, -3, 1, 1, -3, -3, -3, 3, 3, -1]))
        frame2 = vdif.VDIFFrame.frombytes(frame.tobytes())
        assert frame2 == frame
        frame3 = vdif.VDIFFrame.fromdata(frame.data, frame.header)
        assert frame3 == frame

    def test_frameset(self):
        with vdif.open('vlba.m5a', 'rb') as fh:
            header = vdif.VDIFHeader.fromfile(fh)
            fh.seek(0)
            frameset = fh.read_frameset()
        assert len(frameset.frames) == 8
        assert frameset.samples_per_frame == 20000
        assert frameset.nchan == 1
        assert frameset.shape == (8, 20000, 1)
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

    def test_filestreamer(self):
        with open('vlba.m5a', 'rb') as fh:
            header = vdif.VDIFHeader.fromfile(fh)

        with vdif.open('vlba.m5a', 'rs') as fh:
            assert header == fh.header0
            record = fh.read(12)
            assert fh.offset == 12

        assert record.shape == (12, 8)
        assert np.all(record.astype(int)[:, 0] ==
                      np.array([-1, -1, 3, -1, 1, -1, 3, -1, 1, 3, -1, 1]))


class TestVDIFMark5B(object):
    """Test VDIF frame containing Mark5B data (EDV 0xab)."""

    def test_header(self):
        with open('sample.m5b', 'rb') as fh:
            m5h = mark5b.Mark5BHeader.fromfile(fh, Time('2014-06-01').mjd)
            m5pl = mark5b.Mark5BPayload.fromfile(fh, nchan=8, bps=2)
        header = vdif.VDIFHeader.from_mark5b_header(m5h, nchan=m5pl.nchan,
                                                    bps=m5pl.bps)
        assert all(m5h[key] == header[key] for key in m5h.keys())
        assert header.time == m5h.time
        assert header.nchan == 8
        assert header.bps == 2
        assert not header['complex_data']
        assert header.framesize == 10032
        assert header.size == 32
        assert header.payloadsize == m5h.payloadsize
        assert header.samples_per_frame == 10000 * 8 // m5pl.bps // m5pl.nchan

    def test_payload(self):
        with open('sample.m5b', 'rb') as fh:
            m5h = mark5b.Mark5BHeader.fromfile(fh, Time('2014-06-01').mjd)
            m5pl = mark5b.Mark5BPayload.fromfile(fh, nchan=8, bps=2)
        header = vdif.VDIFHeader.from_mark5b_header(m5h, nchan=m5pl.nchan,
                                                    bps=m5pl.bps)
        payload = vdif.VDIFPayload(m5pl.words, header)
        assert np.all(payload.words == m5pl.words)
        assert np.all(payload.data == m5pl.data)
        payload2 = vdif.VDIFPayload.fromdata(m5pl.data, header)
        assert np.all(payload2.words == m5pl.words)
        assert np.all(payload2.data == m5pl.data)

    def test_frame(self):
        with mark5b.open('sample.m5b', 'rb') as fh:
            m5f = fh.read_frame(nchan=8, bps=2, ref_mjd=57000.)
        frame = vdif.VDIFFrame.from_mark5b_frame(m5f)
        assert frame.size == 10032
        assert frame.shape == (5000, 8)
        assert np.all(frame.data == m5f.data)
