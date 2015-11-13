import io
import os
import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.tests.helper import pytest
from .. import vdif
from .. import mark4
from .. import mark5b


SAMPLE_M4 = os.path.join(os.path.dirname(__file__), 'sample.m4')
SAMPLE_M5B = os.path.join(os.path.dirname(__file__), 'sample.m5b')
SAMPLE_VDIF = os.path.join(os.path.dirname(__file__), 'sample.vdif')


class TestVDIFMark5B(object):
    """Simplest conversion: VDIF frame containing Mark5B data (EDV 0xab)."""

    def test_header(self):
        with open(SAMPLE_M5B, 'rb') as fh:
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
        with open(SAMPLE_M5B, 'rb') as fh:
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
        header2 = header.copy()
        header2['complex_data'] = True
        with pytest.raises(ValueError):
            vdif.VDIFPayload(m5pl.words, header2)

    def test_frame(self):
        with mark5b.open(SAMPLE_M5B, 'rb') as fh:
            m5f = fh.read_frame(nchan=8, bps=2, ref_mjd=57000.)
        frame = vdif.VDIFFrame.from_mark5b_frame(m5f)
        assert frame.size == 10032
        assert frame.shape == (5000, 8)
        assert np.all(frame.data == m5f.data)


class TestMark5BToVDIF3(object):
    """Real conversion: Mark5B to VDIF EDV 3, and back to Mark5B"""

    def test_header(self):
        with open(SAMPLE_M5B, 'rb') as fh:
            m5h = mark5b.Mark5BHeader.fromfile(fh, Time('2014-06-01').mjd)
            m5pl = mark5b.Mark5BPayload.fromfile(fh, nchan=8, bps=2)
        # check that we have enough information to create VDIF EDV 3 header.
        header = vdif.VDIFHeader.fromvalues(
            edv=3, bps=m5pl.bps, nchan=1, station='WB', time=m5h.time,
            bandwidth=16.*u.MHz, complex_data=False)
        assert header.time == m5h.time

    def test_stream(self):
        with mark5b.open(SAMPLE_M5B, 'rs', nchan=8, bps=2, ref_mjd=57000,
                         sample_rate=32.*u.MHz) as fr:
            m5h = fr.header0
            header = vdif.VDIFHeader.fromvalues(
                edv=3, bps=fr.bps, nchan=1, station='WB', time=m5h.time,
                bandwidth=16.*u.MHz, complex_data=False)
            data = fr.read(20000)  # enough to fill one EDV3 frame.
            time1 = fr.tell(unit='time')

        with io.BytesIO() as s, vdif.open(s, 'ws', nthread=data.shape[1],
                                          header=header) as fw:
            assert (fw.tell(unit='time') - m5h.time) < 2. * u.ns
            fw.write(data)
            assert (fw.tell(unit='time') - time1) < 2. * u.ns
            fw.fh_raw.flush()
            s.seek(0)
            with mark5b.open(SAMPLE_M5B, 'rs', nchan=8, bps=2, ref_mjd=57000,
                             sample_rate=32.*u.MHz) as fm, vdif.open(
                                 s, 'rs') as fv:
                assert fm.header0.time == fv.header0.time
                dm = fm.read(20000)
                dv = fv.read(20000)
                assert np.all(dm == dv)
                assert fm.offset == fv.offset
                assert fm.tell(unit='time') == fv.tell(unit='time')

                # Convert VDIF file back to Mark 5B, and check byte-for-byte.
                hv = fv.header0
                hm = fm.header0
                with io.BytesIO() as s2, mark5b.open(
                        s2, 'ws', nchan=dv.shape[1], bps=hv.bps, time=hv.time,
                        sample_rate=hv.bandwidth*2, user=hm['user'],
                        internal_tvg=hm['internal_tvg']) as fw:
                    fw.write(dv)
                    number_of_bytes = s2.tell()
                    fm_raw = fm.fh_raw
                    assert number_of_bytes == fm_raw.tell()
                    s2.seek(0)
                    fm_raw.seek(0)
                    orig_bytes = fm_raw.read(number_of_bytes)
                    conv_bytes = s2.read(number_of_bytes)
                    assert orig_bytes == conv_bytes


class TestVDIF3ToMark5B(object):
    """Real conversion: VDIF EDV 3 to Mark5B."""

    def test_header(self):
        with open(SAMPLE_VDIF, 'rb') as fh:
            vh = vdif.VDIFHeader.fromfile(fh)

        header = mark5b.Mark5BHeader.fromvalues(time=vh.time)
        assert header.time == vh.time

    def test_stream(self):
        with vdif.open(SAMPLE_VDIF, 'rs') as fr:
            vh = fr.header0
            data = fr.read(20000)  # enough to fill two Mark 5B frames.

        with io.BytesIO() as s, mark5b.open(s, 'ws', nchan=data.shape[1],
                                            bps=vh.bps, time=vh.time,
                                            sample_rate=vh.bandwidth*2) as fw:
            fw.write(data)
            fw.fh_raw.flush()
            s.seek(0)
            with vdif.open(SAMPLE_VDIF, 'rs') as fv, mark5b.open(
                    s, 'rs', nchan=8, bps=2, ref_mjd=57000,
                    sample_rate=32.*u.MHz) as fm:
                assert fv.header0.time == fm.header0.time
                dv = fv.read(20000)
                dm = fm.read(20000)
                assert np.all(dm == dv)
                assert fm.offset == fv.offset
                assert fm.tell(unit='time') == fv.tell(unit='time')


class TestMark4ToVDIF1(object):
    """Real conversion: Mark 4 to VDIF EDV 1, and back to Mark 4.

    Here, need to use a VDIF format with a flexible size, since we want
    to create invalid frames corresponding to the pieces of data overwritten
    by the Mark 4 header.
    """

    def test_header(self):
        with open(SAMPLE_M4, 'rb') as fh:
            fh.seek(0xa88)
            m4h = mark4.Mark4Header.fromfile(fh, ntrack=64, decade=2010)
        # check that we have enough information to create VDIF EDV 1 header.
        header = vdif.VDIFHeader.fromvalues(
            edv=1, bps=m4h.bps, nchan=1, station='Ar', time=m4h.time,
            bandwidth=16.*u.MHz, payloadsize=640*2//8, complex_data=False)
        assert abs(header.time - m4h.time) < 2. * u.ns

    def test_stream(self):
        with mark4.open(SAMPLE_M4, 'rs', ntrack=64, decade=2010,
                        sample_rate=32.*u.MHz) as fr:
            m4h = fr.header0
            header = vdif.VDIFHeader.fromvalues(
                edv=1, bps=m4h.bps, nchan=1, station='Ar', time=m4h.time,
                bandwidth=16.*u.MHz, payloadsize=640*2//8, complex_data=False)
            assert abs(header.time - m4h.time) < 2. * u.ns
            data = fr.read(80000)  # full Mark 4 frame
            time1 = fr.tell(unit='time')

        with io.BytesIO() as s, vdif.open(s, 'ws', nthread=data.shape[1],
                                          header=header) as fw:
            assert (fw.tell(unit='time') - header.time) < 2. * u.ns
            # Write first VDIF frame, matching Mark 4 Header, hence invalid.
            fw.write(data[:160], invalid_data=True)
            # Write remaining VDIF frames, with valid data.
            fw.write(data[160:])
            assert (fw.tell(unit='time') - time1) < 2. * u.ns
            fw.fh_raw.flush()
            s.seek(0)
            with mark4.open(SAMPLE_M4, 'rs', ntrack=64, decade=2010,
                            sample_rate=32.*u.MHz) as fm, vdif.open(
                                s, 'rs') as fv:
                assert abs(fm.header0.time - fv.header0.time) < 2. * u.ns
                dm = fm.read(80000)
                dv = fv.read(80000)
                assert np.all(dm == dv)
                assert fm.offset == fv.offset
                assert (abs(fm.tell(unit='time') - fv.tell(unit='time')) <
                        2.*u.ns)

                # Convert VDIF file back to Mark 4, and check byte-for-byte.
                hv = fv.header0
                with io.BytesIO() as s2, mark4.open(
                        s2, 'ws', sample_rate=hv.bandwidth*2,
                        time=hv.time, ntrack=64, bps=2, fanout=4,
                        bcd_headstack1=0x3344, bcd_headstack2=0x1122,
                        lsb_output=True, system_id=108) as fw:
                    fw.write(dv)
                    number_of_bytes = s2.tell()
                    fm_raw = fm.fh_raw
                    assert number_of_bytes == fm_raw.tell() - 0xa88
                    s2.seek(0)
                    fm_raw.seek(0xa88)
                    orig_bytes = fm_raw.read(number_of_bytes)
                    conv_bytes = s2.read(number_of_bytes)
                    assert orig_bytes == conv_bytes
