import io
import numpy as np
from astropy import units as u
from astropy.time import Time
from .. import vdif
from .. import mark5b


class TestVDIFMark5B(object):
    """Simplest conversion: VDIF frame containing Mark5B data (EDV 0xab)."""

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


class TestMark5BToVDIF3(object):
    """Real conversion: Mark5B to VDIF EDV 3."""

    def test_header(self):
        with open('sample.m5b', 'rb') as fh:
            m5h = mark5b.Mark5BHeader.fromfile(fh, Time('2014-06-01').mjd)
            m5pl = mark5b.Mark5BPayload.fromfile(fh, nchan=8, bps=2)
        # check that we have enough information to create VDIF EDV 3 header.
        header = vdif.VDIFHeader.fromvalues(
            edv=3, bps=m5pl.bps, nchan=1, station='WB', time=m5h.time,
            bandwidth=16.*u.MHz, complex_data=False)
        assert header.time == m5h.time

    def test_stream(self):
        with mark5b.open('sample.m5b', 'rs', nchan=8, bps=2, ref_mjd=57000,
                         sample_rate=32.*u.MHz) as fr:
            m5h = fr.header0
            header = vdif.VDIFHeader.fromvalues(
                edv=3, bps=fr.bps, nchan=1, station='WB', time=m5h.time,
                bandwidth=16.*u.MHz, complex_data=False)
            data = fr.read(20000)  # enough to fill one EDV3 frame.

        with io.BytesIO() as s, vdif.open(s, 'ws', nthread=data.shape[1],
                                          header=header) as fw:
            fw.write(data)
            fw.fh_raw.flush()
            s.seek(0)
            with mark5b.open('sample.m5b', 'rs', nchan=8, bps=2, ref_mjd=57000,
                             sample_rate=32.*u.MHz) as fm, vdif.open(
                                 s, 'rs') as fv:
                assert fm.header0.time == fv.header0.time
                dm = fm.read(20000)
                dv = fv.read(20000)
                assert np.all(dm == dv)
                assert fm.offset == fv.offset
                assert fm.tell(unit='time') == fv.tell(unit='time')
