# Licensed under the GPLv3 - see LICENSE
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time
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
            # Start time kiloday is needed for Mark 5B to calculate time.
            m5h1 = mark5b.Mark5BHeader.fromfile(fh, kday=56000)
            # For the payload, pass in how data is encoded.
            m5pl = mark5b.Mark5BPayload.fromfile(fh, nchan=8, bps=2)
            # A not-at-the-start header for checking times.
            m5h2 = mark5b.Mark5BHeader.fromfile(fh, kday=56000)
        # Create VDIF headers based on both the Mark 5B header and payload.
        header1 = vdif.VDIFHeader.from_mark5b_header(
            m5h1, nchan=m5pl.sample_shape.nchan, bps=m5pl.bps)
        header2 = vdif.VDIFHeader.from_mark5b_header(
            m5h2, nchan=m5pl.sample_shape.nchan, bps=m5pl.bps)
        for i, (m5h, header) in enumerate(((m5h1, header1), (m5h2, header2))):
            assert m5h['frame_nr'] == i
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
            assert header.frame_nbytes == 10032
            assert header.nbytes == 32
            assert header.payload_nbytes == m5h.payload_nbytes
            assert (header.samples_per_frame ==
                    10000 * 8 // m5pl.bps // m5pl.sample_shape.nchan)

        # Check that we can handle > 512 Mbps sampling rate.
        header3 = vdif.VDIFHeader.from_mark5b_header(
            m5h2, nchan=m5pl.sample_shape.nchan, bps=m5pl.bps,
            sample_rate=64*u.MHz)
        assert header3.time == header2.time
        assert header3['frame_nr'] == m5h2['frame_nr']

        # A copy might remove any `kday` keywords set, but should still work
        # (Regression test for #34)
        header_copy = header2.copy()
        assert header_copy == header2
        header_copy.verify()
        # But it should not remove `kday` to start with (#35)
        assert header_copy.kday == header2.kday
        # May as well check that with a corrupt 'bcd_fraction' we can still
        # get the right time using the frame number.
        header_copy['bcd_fraction'] = 0
        # This is common enough that we should not fail verification.
        header_copy.verify()
        # However, it should also cause just getting the time to fail
        # unless we pass in a frame rate.
        with pytest.raises(ValueError):
            header_copy.time
        frame_rate = 32. * u.MHz / header.samples_per_frame
        assert abs(header_copy.get_time(frame_rate=frame_rate) -
                   m5h2.time) < 1.*u.ns

    def test_payload(self):
        """Check Mark 5B payloads can used in a Mark5B VDIF payload."""
        # Get Mark 5B header, payload, and construct VDIF header, as above.
        with open(SAMPLE_M5B, 'rb') as fh:
            m5h = mark5b.Mark5BHeader.fromfile(fh, kday=56000)
            m5pl = mark5b.Mark5BPayload.fromfile(fh, nchan=8, bps=2)
        header = vdif.VDIFHeader.from_mark5b_header(
            m5h, nchan=m5pl.sample_shape.nchan, bps=m5pl.bps)
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
        with mark5b.open(SAMPLE_M5B, 'rb', ref_time=Time(57000, format='mjd'),
                         nchan=8, bps=2) as fh:
            # pick second frame just to be different from header checks above.
            fh.seek(10016)
            m5f = fh.read_frame()

        assert m5f['frame_nr'] == 1
        frame = vdif.VDIFFrame.from_mark5b_frame(m5f)
        assert frame.nbytes == 10032
        assert frame.shape == (5000, 8)
        assert np.all(frame.data == m5f.data)
        assert frame.time == m5f.time

    def test_stream(self):
        """Check we can encode a whole stream."""


class TestVDIF0VDIF1(object):
    """Conversion between EDV versions."""

    def test_stream(self, tmpdir):
        with vdif.open(SAMPLE_MWA, 'rs', sample_rate=1.28*u.MHz) as f0:
            h0 = f0.header0
            d0 = f0.read(1024)
            kwargs = dict(h0)
            kwargs['edv'] = 1
            fl = str(tmpdir.join('test1.vdif'))
            with vdif.open(fl, 'ws', sample_rate=1.28*u.MHz, **kwargs) as f1w:
                h1w = f1w.header0
                assert list(h1w.words[:4]) == list(h0.words[:4])
                assert h1w.sample_rate == 1.28*u.MHz
                f1w.write(d0)

            with vdif.open(fl, 'rs') as f1r:
                h1r = f1r.header0
                d1r = f1r.read(1024)
                assert h1r.words[:4] == h0.words[:4]
                assert h1w.sample_rate == 1.28*u.MHz
                assert np.all(d1r == d0)


class TestMark5BToVDIF3(object):
    """Real conversion: Mark5B to VDIF EDV 3, and back to Mark5B"""

    def test_header(self):
        with open(SAMPLE_M5B, 'rb') as fh:
            m5h = mark5b.Mark5BHeader.fromfile(fh, kday=56000)
            m5pl = mark5b.Mark5BPayload.fromfile(fh, nchan=8, bps=2)
        # check that we have enough information to create VDIF EDV 3 header.
        header = vdif.VDIFHeader.fromvalues(
            edv=3, bps=m5pl.bps, nchan=1, station='WB', time=m5h.time,
            sample_rate=32.*u.MHz, complex_data=False)
        assert header.time == m5h.time

    def test_stream(self, tmpdir):
        """Convert Mark 5B data stream to VDIF."""
        # Here, we need to give how the data is encoded, since the data do not
        # self-describe this.  Furthermore, we need to pass in a rough time,
        # and the rate at which samples were taken, so that absolute times can
        # be calculated.
        with mark5b.open(SAMPLE_M5B, 'rs', sample_rate=32.*u.MHz, kday=56000,
                         nchan=8, bps=2) as fr:
            m5h = fr.header0
            # create VDIF header from Mark 5B stream information.
            header = vdif.VDIFHeader.fromvalues(
                edv=3, bps=fr.bps, nchan=1, station='WB', time=m5h.time,
                sample_rate=32.*u.MHz, complex_data=False)
            data = fr.read(20000)  # enough to fill one EDV3 frame.
            time1 = fr.tell(unit='time')

        # Get a file name in our temporary testing directory.
        vdif_file = str(tmpdir.join('converted.vdif'))
        # create and fill vdif file with converted data.
        with vdif.open(vdif_file, 'ws', header0=header,
                       nthread=data.shape[1]) as fw:
            assert (fw.tell(unit='time') - m5h.time) < 2. * u.ns
            fw.write(data)
            assert (fw.tell(unit='time') - time1) < 2. * u.ns

        # Check two files contain same information.
        with mark5b.open(SAMPLE_M5B, 'rs', sample_rate=32.*u.MHz, kday=56000,
                         nchan=8, bps=2) as fm, vdif.open(vdif_file,
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
            with mark5b.open(mark5b_new_file, 'ws', sample_rate=hv.sample_rate,
                             nchan=dv.shape[1], bps=hv.bps,
                             time=hv.time, user=hm['user'],
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

    def test_stream(self, tmpdir):
        with vdif.open(SAMPLE_VDIF, 'rs') as fr:
            vh = fr.header0
            data = fr.read(20000)  # enough to fill two Mark 5B frames.

        fl = str(tmpdir.join('test.m5b'))
        with mark5b.open(fl, 'ws', sample_rate=vh.sample_rate,
                         nchan=data.shape[1], bps=vh.bps, time=vh.time) as fw:
            fw.write(data)

        with vdif.open(SAMPLE_VDIF, 'rs') as fv, mark5b.open(
                fl, 'rs', sample_rate=32.*u.MHz,
                ref_time=Time(57000, format='mjd'), nchan=8, bps=2) as fm:
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
        # Check that we have enough information to create VDIF EDV 1 header.
        header = vdif.VDIFHeader.fromvalues(
            edv=1, bps=m4h.bps, nchan=1, station='Ar', time=m4h.time,
            sample_rate=32.*u.MHz, payload_nbytes=640*2//8, complex_data=False)
        assert abs(header.time - m4h.time) < 2. * u.ns

    def test_stream(self, tmpdir):
        with mark4.open(SAMPLE_M4, 'rs', sample_rate=32.*u.MHz,
                        ntrack=64, decade=2010) as fr:
            m4header0 = fr.header0
            start_time = fr.start_time
            vheader0 = vdif.VDIFHeader.fromvalues(
                edv=1, bps=m4header0.bps, nchan=1, station='Ar',
                time=start_time, sample_rate=32.*u.MHz,
                payload_nbytes=640*2//8, complex_data=False)
            assert abs(vheader0.time - start_time) < 2. * u.ns
            data = fr.read(80000)  # full Mark 4 frame
            offset1 = fr.tell()
            time1 = fr.tell(unit='time')
            number_of_bytes = fr.fh_raw.tell() - 0xa88

        with open(SAMPLE_M4, 'rb') as fh:
            fh.seek(0xa88)
            orig_bytes = fh.read(number_of_bytes)

        fl = str(tmpdir.join('test.vdif'))
        with vdif.open(fl, 'ws', header0=vheader0,
                       nthread=data.shape[1]) as fw:
            assert (fw.tell(unit='time') - start_time) < 2. * u.ns
            # Write first VDIF frame, matching Mark 4 Header, hence invalid.
            fw.write(data[:160], valid=False)
            # Write remaining VDIF frames, with valid data.
            fw.write(data[160:])
            assert (fw.tell(unit='time') - time1) < 2. * u.ns

        with vdif.open(fl, 'rs') as fv:
            assert abs(fv.header0.time - start_time) < 2. * u.ns
            expected = vheader0.copy()
            expected['invalid_data'] = True
            assert fv.header0 == expected
            dv = fv.read(80000)
            assert np.all(dv == data)
            assert fv.offset == offset1
            assert abs(fv.tell(unit='time') - time1) < 2.*u.ns

        # Convert VDIF file back to Mark 4, and check byte-for-byte.
        fl2 = str(tmpdir.join('test.m4'))
        with mark4.open(fl2, 'ws', sample_rate=vheader0.sample_rate,
                        ntrack=64, bps=2, fanout=4, time=vheader0.time,
                        system_id=108) as fw:
            fw.write(dv)

        with open(fl2, 'rb') as fh:
            conv_bytes = fh.read()
            assert len(conv_bytes) == len(conv_bytes)
            assert orig_bytes == conv_bytes


class TestDADAToVDIF1(object):
    """Real conversion: DADA to VDIF EDV 1, and back to DADA.

    Here, we use a VDIF format with a flexible size so it is easier to fit
    the dada file inside the VDIF one.
    """

    def get_vdif_header(self, header):
        return vdif.VDIFHeader.fromvalues(
            edv=1, time=header.time, sample_rate=header.sample_rate,
            bps=header.bps, nchan=header['NCHAN'],
            complex_data=header.complex_data,
            payload_nbytes=header.payload_nbytes // 2,
            station=header['TELESCOPE'][:2])

    def get_vdif_data(self, dada_data):
        return (dada_data + 0.5 + 0.5j) / EIGHT_BIT_1_SIGMA

    def get_dada_data(self, vdif_data):
        return vdif_data * EIGHT_BIT_1_SIGMA - 0.5 - 0.5j

    def test_header(self):
        with open(SAMPLE_DADA, 'rb') as fh:
            ddh = dada.DADAHeader.fromfile(fh)
        # Check that we have enough information to create VDIF EDV 1 header.
        header = self.get_vdif_header(ddh)
        assert abs(header.time - ddh.time) < 2. * u.ns
        assert header.payload_nbytes == ddh.payload_nbytes // 2

    def test_payload(self):
        with open(SAMPLE_DADA, 'rb') as fh:
            fh.seek(4096)
            ddp = dada.DADAPayload.fromfile(fh, payload_nbytes=64000,
                                            sample_shape=(2, 1),
                                            complex_data=True, bps=8)
        dada_data = ddp.data
        # Check that conversion between scalings works.
        vdif_data = self.get_vdif_data(dada_data)
        assert np.allclose(self.get_dada_data(vdif_data), dada_data)
        # Check that we can create correct payloads.
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
            stop_time = fr.tell(unit='time')

        header = self.get_vdif_header(ddh)
        data = self.get_vdif_data(dada_data)
        assert abs(header.time - ddh.time) < 2. * u.ns
        vdif_file = str(tmpdir.join('converted_dada.vdif'))
        with vdif.open(vdif_file, 'ws', header0=header,
                       nthread=data.shape[1]) as fw:
            assert (fw.tell(unit='time') - header.time) < 2. * u.ns
            # Write all data in since frameset, made of two frames.
            fw.write(data)
            assert (fw.tell(unit='time') - stop_time) < 2. * u.ns
            assert fw.offset == offset1

        with vdif.open(vdif_file, 'rs') as fv:
            assert abs(fv.header0.time - ddh.time) < 2. * u.ns
            dv = fv.read()
            assert fv.offset == offset1
            assert np.abs(fv.tell(unit='time') - stop_time) < 2.*u.ns
            vh = fv.header0
            vnthread = fv.sample_shape.nthread
        assert np.allclose(dv, data)

        # Convert VDIF file back to DADA.
        dada_file = str(tmpdir.join('reconverted.dada'))
        dv_data = self.get_dada_data(dv)
        assert np.allclose(dv_data, dada_data)
        with dada.open(dada_file, 'ws', sample_rate=vh.sample_rate,
                       time=vh.time, npol=vnthread, bps=vh.bps,
                       payload_nbytes=vh.payload_nbytes*2, nchan=vh.nchan,
                       telescope=vh.station,
                       complex_data=vh['complex_data']) as fw:
            new_header = fw.header0
            fw.write(dv_data)

        assert self.get_vdif_header(new_header) == vh
        with dada.open(dada_file, 'rs') as fh:
            header = fh.header0
            new_dada_data = fh.read()
        assert header == new_header
        assert self.get_vdif_header(header) == vh
        assert np.allclose(new_dada_data, dada_data)
