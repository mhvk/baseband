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
