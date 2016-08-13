# Licensed under the GPLv3 - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import io
import numpy as np
from astropy.tests.helper import pytest
from .. import dada


SAMPLE_FILE = os.path.join(os.path.dirname(__file__), 'sample.dada')


class TestDADA(object):
    def test_header(self):
        with open(SAMPLE_FILE, 'rb') as fh:
            header = dada.DADAHeader.fromfile(fh)
            assert header.size == 4096
            assert fh.tell() == 4096
        assert header['NDIM'] == 2
        assert header['NCHAN'] == 1
        assert header['UTC_START'] == '2013-07-02-01:37:40'
        assert header['OBS_OFFSET'] == 6400000000  # 100 s
        assert header.time.isot == '2013-07-02T01:39:20.000'
        assert header.framesize == 64000 + 4096
        assert header.payloadsize == 64000
        assert header.mutable is False
        with io.BytesIO() as s:
            header.tofile(s)
            assert s.tell() == header.size
            s.seek(0)
            header2 = dada.DADAHeader.fromfile(s)
            assert s.tell() == header.size
        assert header2 == header
        assert header2.mutable is False
        # Note that this is not guaranteed to preserve order!
        header3 = dada.DADAHeader.fromkeys(**header)
        assert header3 == header
        assert header3.mutable is True
        # # Try initialising with properties instead of keywords.
        # Here, we first just try the start time.
        header4 = dada.DADAHeader.fromvalues(
            time0=header.time0, offset=header.time-header.time0,
            bps=header.bps, complex_data=header.complex_data,
            bandwidth=header.bandwidth, sideband=header.sideband,
            samples_per_frame=header.samples_per_frame,
            npol=header['NPOL'], nchan=header['NCHAN'],
            source=header['SOURCE'], ra=header['RA'], dec=header['DEC'],
            telescope=header['TELESCOPE'], instrument=header['INSTRUMENT'],
            receiver=header['RECEIVER'], freq=header['FREQ'],
            pic_version=header['PIC_VERSION'])
        assert header4 == header
        assert header4.mutable is True
        # And now try both start time and time of observation.
        header5 = dada.DADAHeader.fromvalues(
            offset=header.offset, time=header.time,
            bps=header.bps, complex_data=header.complex_data,
            bandwidth=header.bandwidth, sideband=header.sideband,
            samples_per_frame=header.samples_per_frame,
            npol=header['NPOL'], nchan=header['NCHAN'],
            source=header['SOURCE'], ra=header['RA'], dec=header['DEC'],
            telescope=header['TELESCOPE'], instrument=header['INSTRUMENT'],
            receiver=header['RECEIVER'], freq=header['FREQ'],
            pic_version=header['PIC_VERSION'])
        assert header5 == header
        # Check repr can be used to instantiate header
        header6 = eval('dada.' + repr(header))
        assert header6 == header
        # repr includes the comments
        assert header6.comments == header.comments
        # Therefore repr should be identical too.
        assert repr(header6) == repr(header)
        # Check instantiation via tuple
        header7 = dada.DADAHeader(((key, (header[key], header.comments[key]))
                                   for key in header))
        assert header7 == header
        assert header7.comments == header.comments
        # Check copying
        header8 = header.copy()
        assert header8 == header
        assert header8.mutable is True
        assert header8.comments == header.comments

    def test_payload(self):
        with open(SAMPLE_FILE, 'rb') as fh:
            fh.seek(4096)  # skip header
            payload = dada.DADAPayload.fromfile(fh, payloadsize=64000, bps=8,
                                                complex_data=True,
                                                sample_shape=(2,))
        assert payload.size == 64000
        assert payload.shape == (16000, 2)
        assert payload.dtype == np.complex64
        data = payload.data
        assert np.all(data[:3] ==
                      np.array([[-38.-38.j, -38.-38.j],
                                [-38.-38.j, -40.+0.j],
                                [-105.+60.j, 85.-15.j]], dtype=np.complex64))

        with io.BytesIO() as s:
            payload.tofile(s)
            s.seek(0)
            payload2 = dada.DADAPayload.fromfile(s, payloadsize=64000, bps=8,
                                                 complex_data=True,
                                                 sample_shape=(2,))
            assert payload2 == payload
            with pytest.raises(EOFError):
                # Too few bytes.
                s.seek(100)
                dada.DADAPayload.fromfile(s, payloadsize=64000, bps=8,
                                          complex_data=True,
                                          sample_shape=(2,))
        payload3 = dada.DADAPayload.fromdata(data, bps=8)
        assert payload3 == payload

    def test_filestreamer(self):
        with open(SAMPLE_FILE, 'rb') as fh:
            header = dada.DADAHeader.fromfile(fh)
            payload = dada.DADAPayload.fromfile(fh, payloadsize=64000, bps=8,
                                                complex_data=True,
                                                sample_shape=(2,))

        with dada.open(SAMPLE_FILE, 'rs') as fh:
            assert header == fh.header0
            assert fh.fh_raw.tell() == header.size
            assert fh.size == 16000
            record = fh.read(12)
            assert fh.tell() == 12
            fh.seek(10000)
            record2 = fh.read(2)
            assert fh.tell() == 10002
            assert fh.fh_raw.tell() == header.size + 10002 * 4
        #     assert np.abs(fh.tell(unit='time') -
        #                   (fh.time0 + 10002 / (32*u.MHz))) < 1. * u.ns
        #     fh.seek(fh.time0 + 1000 / (32*u.MHz))
        #     assert fh.tell() == 1000

        # assert header1['frame_nr'] == 3
        # assert header1['user'] == header['user']
        # assert header1['bcd_jday'] == header['bcd_jday']
        # assert header1['bcd_seconds'] == header['bcd_seconds']
        # assert header1['bcd_fraction'] == 4
        # assert (round((1./((header1.time-header.time)/3.)).to(u.Hz).value) ==
        #         6400)
        # assert record.shape == (12, 8)
        # assert np.all(record.astype(int)[:3] ==
        #               np.array([[-3, -1, +1, -1, +3, -3, -3, +3],
        #                         [-3, +3, -1, +3, -1, -1, -1, +1],
        #                         [+3, -1, +3, +3, +1, -1, +3, -1]]))
        # assert record2.shape == (2, 8)
        # assert np.all(record2.astype(int) ==
        #               np.array([[-1, -1, -1, +3, +3, -3, +3, -1],
        #                         [-1, +1, -3, +3, -3, +1, +3, +1]]))
        assert record.shape == (12, 2) and record.dtype == np.complex64
        assert np.all(record == payload.data[:12])
        assert np.all(record2 == payload.data[10000:10002])
