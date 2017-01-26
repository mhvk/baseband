import io
import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.tests.helper import pytest
from .. import vdif
from .. import mark4
from .. import mark5b
from .. import dada
from ..vlbi_base.encoding import EIGHT_BIT_1_SIGMA
from ..data import (SAMPLE_MARK4 as SAMPLE_M4, SAMPLE_MARK5B as SAMPLE_M5B,
                    SAMPLE_VDIF, SAMPLE_MWA_VDIF as SAMPLE_MWA, SAMPLE_DADA)


class TestVDIFMark5B(object):
    """Simplest conversion: VDIF frame containing Mark5B data (EDV 0xab)."""

    def test_header(self):
        """Check Mark 5B header information can be stored in a VDIF header."""
        with open(SAMPLE_M5B, 'rb') as fh:
            # A rough start time is needed for Mark 5B to calculate time.
            m5h = mark5b.Mark5BHeader.fromfile(fh, Time('2014-06-01').mjd)
            # For the payload, pass in how data is encoded.
            m5pl = mark5b.Mark5BPayload.fromfile(fh, nchan=8, bps=2)
        # Create a VDIF header based on both the Mark 5B header and payload.
        header = vdif.VDIFHeader.from_mark5b_header(m5h, nchan=m5pl.nchan,
                                                    bps=m5pl.bps)
        # Check all direct information is set correctly.
        assert all(m5h[key] == header[key] for key in m5h.keys())
        assert header['mark5b_frame_nr'] == m5h['frame_nr']
        assert header.kday == m5h.kday
        # As well as the time calculated from the header information.
        assert header.time == m5h.time
        # Check information on the payload is also correct.
        assert header.nchan == 8
        assert header.bps == 2
        assert not header['complex_data']
        assert header.framesize == 10032
        assert header.size == 32
        assert header.payloadsize == m5h.payloadsize
        assert header.samples_per_frame == 10000 * 8 // m5pl.bps // m5pl.nchan
        # A copy might remove any `kday` keywords set, but should still work
        # (Regression test for #34)
        header1 = header.copy()
        header1.verify()
        # But it should not remove `kday` to start with (#35)
        assert header1.kday == header.kday

    def test_payload(self):
        """Check Mark 5B payloads can used in a Mark5B VDIF payload."""
        # Get Mark 5B header, payload, and construct VDIF header, as above.
        with open(SAMPLE_M5B, 'rb') as fh:
            m5h = mark5b.Mark5BHeader.fromfile(fh, Time('2014-06-01').mjd)
            m5pl = mark5b.Mark5BPayload.fromfile(fh, nchan=8, bps=2)
        header = vdif.VDIFHeader.from_mark5b_header(m5h, nchan=m5pl.nchan,
                                                    bps=m5pl.bps)
        # Create VDIF payload from the Mark 5B encoded payload.
        payload = vdif.VDIFPayload(m5pl.words, header)
        # Check that the payload (i.e., encoded data) is the same.
        assert np.all(payload.words == m5pl.words)
        # And check that if we decode the payload, we get the same result.
        assert np.all(payload.data == m5pl.data)
        # Now construct a VDIF payload from the Mark 5B data, checking that
        # the encoding works correctly too.
        payload2 = vdif.VDIFPayload.fromdata(m5pl.data, header)
        assert np.all(payload2.words == m5pl.words)
        assert np.all(payload2.data == m5pl.data)
        # Mark 5B data cannot complex. Check that this raises an exception.
        header2 = header.copy()
        header2['complex_data'] = True
        with pytest.raises(ValueError):
            vdif.VDIFPayload(m5pl.words, header2)

    def test_frame(self):
        """Check a whole Mark 5B frame can be translated to VDIF."""
        with mark5b.open(SAMPLE_M5B, 'rb') as fh:
            m5f = fh.read_frame(nchan=8, bps=2, ref_mjd=57000.)
        frame = vdif.VDIFFrame.from_mark5b_frame(m5f)
        assert frame.size == 10032
        assert frame.shape == (5000, 8)
        assert np.all(frame.data == m5f.data)


class TestVDIF0VDIF1(object):
    """Conversion between EDV versions."""
    with vdif.open(SAMPLE_MWA, 'rs', sample_rate=1.28*u.MHz) as f0:
        h0 = f0.header0
        d0 = f0.read(1024)
        with io.BytesIO() as s:
            kwargs = dict(h0)
            kwargs['edv'] = 1
            with vdif.open(s, 'ws', frames_per_second=10000, **kwargs) as f1w:
                h1w = f1w.header0
                assert list(h1w.words[:4]) == list(h0.words[:4])
                assert h1w.framerate == 10. * u.kHz
                assert h1w.bandwidth == 1.28 * f0.nchan * u.MHz
                f1w.write(d0)
                f1w.fh_raw.flush()

                s.seek(0)
                with vdif.open(s, 'rs') as f1r:
                    h1r = f1r.header0
                    d1r = f1r.read(1024)
                assert h1r.words[:4] == h0.words[:4]
                assert h1r.framerate == 10. * u.kHz
                assert h1r.bandwidth == 1.28 * f0.nchan * u.MHz
                assert np.all(d1r == d0)


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

    def test_stream(self, tmpdir):
        """Convert Mark 5B data stream to VDIF."""
        # Here, we need to give how the data is encoded, since the data do not
        # self-describe this.  Furthermore, we need to pass in a rough time,
        # and the rate at which samples were taken, so that absolute times can
        # be calculated.
        with mark5b.open(SAMPLE_M5B, 'rs', nchan=8, bps=2, ref_mjd=57000,
                         sample_rate=32.*u.MHz) as fr:
            m5h = fr.header0
            # create VDIF header from Mark 5B stream information.
            header = vdif.VDIFHeader.fromvalues(
                edv=3, bps=fr.bps, nchan=1, station='WB', time=m5h.time,
                bandwidth=16.*u.MHz, complex_data=False)
            data = fr.read(20000)  # enough to fill one EDV3 frame.
            time1 = fr.tell(unit='time')

        # Get a file name in our temporary testing directory.
        vdif_file = str(tmpdir.join('converted.vdif'))
        # create and fill vdif file with converted data.
        with vdif.open(vdif_file, 'ws', nthread=data.shape[1],
                       header=header) as fw:
            assert (fw.tell(unit='time') - m5h.time) < 2. * u.ns
            fw.write(data)
            assert (fw.tell(unit='time') - time1) < 2. * u.ns

        # check two files contain same information.
        with mark5b.open(SAMPLE_M5B, 'rs', nchan=8, bps=2, ref_mjd=57000,
                         sample_rate=32.*u.MHz) as fm, vdif.open(vdif_file,
                                                                 'rs') as fv:
            assert fm.header0.time == fv.header0.time
            dm = fm.read(20000)
            dv = fv.read(20000)
            assert np.all(dm == dv)
            assert fm.offset == fv.offset
            assert fm.tell(unit='time') == fv.tell(unit='time')

            # Convert VDIF file back to Mark 5B
            mark5b_new_file = str(tmpdir.join('reconverted.mark5b'))
            hv = fv.header0
            hm = fm.header0
            # Here, we fill some unimportant Mark 5B header information by
            # hand, so we can compare byte-for-byte.
            with mark5b.open(mark5b_new_file, 'ws', nchan=dv.shape[1],
                             bps=hv.bps, time=hv.time,
                             sample_rate=hv.bandwidth*2, user=hm['user'],
                             internal_tvg=hm['internal_tvg']) as fw:
                fw.write(dv)

        with open(SAMPLE_M5B, 'rb') as fh_orig, open(mark5b_new_file,
                                                     'rb') as fh_new:
            assert fh_orig.read() == fh_new.read()


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


class TestDADAToVDIF1(object):
    """Real conversion: DADA to VDIF EDV 1, and back to DADA.

    Here, we use a VDIF format with a flexible size so it is easier to fit
    the dada file inside the VDIF one.
    """
    def get_vdif_header(self, header):
        return vdif.VDIFHeader.fromvalues(
            edv=1, time=header.time, bandwidth=header.bandwidth,
            bps=header.bps, nchan=header['NCHAN'],
            complex_data=header.complex_data,
            payloadsize=header.payloadsize // 2,
            station=header['TELESCOPE'][:2])

    def get_vdif_data(self, dada_data):
        return (dada_data + 0.5 + 0.5j) / EIGHT_BIT_1_SIGMA

    def get_dada_data(self, vdif_data):
        return vdif_data * EIGHT_BIT_1_SIGMA - 0.5 - 0.5j

    def test_header(self):
        with open(SAMPLE_DADA, 'rb') as fh:
            ddh = dada.DADAHeader.fromfile(fh)
        # check that we have enough information to create VDIF EDV 1 header.
        header = self.get_vdif_header(ddh)
        assert abs(header.time - ddh.time) < 2. * u.ns
        assert header.payloadsize == ddh.payloadsize // 2

    def test_payload(self):
        with open(SAMPLE_DADA, 'rb') as fh:
            fh.seek(4096)
            ddp = dada.DADAPayload.fromfile(fh, sample_shape=(2, 1),
                                            complex_data=True, bps=8,
                                            payloadsize=64000)
        dada_data = ddp.data
        # check that conversion between scalings works.
        vdif_data = self.get_vdif_data(dada_data)
        assert np.allclose(self.get_dada_data(vdif_data), dada_data)
        # check that we can create correct payloads
        vdif_payload0 = vdif.VDIFPayload.fromdata(vdif_data[:, 0, :], bps=8)
        vdif_payload1 = vdif.VDIFPayload.fromdata(vdif_data[:, 1, :], bps=8)
        vd0, vd1 = vdif_payload0.data, vdif_payload1.data
        assert np.allclose(vd0, vdif_data[:, 0, :])
        assert np.allclose(vd1, vdif_data[:, 1, :])
        vd = np.zeros((vd0.shape[0], 2, vd0.shape[1]), vd0.dtype)
        vd[:, 0] = vd0
        vd[:, 1] = vd1
        dd_new = self.get_dada_data(vd)
        ddp2 = dada.DADAPayload.fromdata(dd_new, bps=8)
        assert ddp2 == ddp

    def test_stream(self, tmpdir):
        with dada.open(SAMPLE_DADA, 'rs') as fr:
            ddh = fr.header0
            dada_data = fr.read()
            offset1 = fr.tell()
            time1 = fr.tell(unit='time')

        header = self.get_vdif_header(ddh)
        data = self.get_vdif_data(dada_data)
        assert abs(header.time - ddh.time) < 2. * u.ns
        vdif_file = str(tmpdir.join('converted_dada.vdif'))
        with vdif.open(vdif_file, 'ws', nthread=data.shape[1],
                       header=header) as fw:
            assert (fw.tell(unit='time') - header.time) < 2. * u.ns
            # Write all data in since frameset, made of two frames.
            fw.write(data)
            assert (fw.tell(unit='time') - time1) < 2. * u.ns
            assert fw.offset == offset1

        with vdif.open(vdif_file, 'rs') as fv:
            assert abs(fv.header0.time - ddh.time) < 2. * u.ns
            dv = fv.read()
            assert fv.offset == offset1
            assert np.abs(fv.tell(unit='time') - time1) < 2.*u.ns
            vh = fv.header0
            vnthread = fv.nthread
        assert np.allclose(dv, data)

        # Convert VDIF file back to DADA.
        dada_file = str(tmpdir.join('reconverted.dada'))
        dv_data = self.get_dada_data(dv)
        assert np.allclose(dv_data, dada_data)
        with dada.open(dada_file, 'ws', bandwidth=vh.bandwidth,
                       time=vh.time, npol=vnthread, bps=vh.bps,
                       payloadsize=vh.payloadsize*2, nchan=vh.nchan,
                       telescope=vh.station,
                       complex_data=vh.complex_data) as fw:
            new_header = fw.header0
            fw.write(dv_data)

        assert self.get_vdif_header(new_header) == vh
        with dada.open(dada_file, 'rs') as fh:
            header = fh.header0
            new_dada_data = fh.read()
        assert header == new_header
        assert self.get_vdif_header(header) == vh
        assert np.allclose(new_dada_data, dada_data)
