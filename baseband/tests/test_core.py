# Licensed under the GPLv3 - see LICENSE
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pytest
from astropy.time import Time
import astropy.units as u

from .. import open as baseband_open, file_info
from ..data import (SAMPLE_MARK4 as SAMPLE_M4, SAMPLE_MARK5B as SAMPLE_M5B,
                    SAMPLE_VDIF, SAMPLE_MWA_VDIF as SAMPLE_MWA, SAMPLE_DADA)


@pytest.mark.parametrize(
    ('sample', 'fmt'),
    ((SAMPLE_M4, 'mark4'),
     (SAMPLE_M5B, 'mark5b'),
     (SAMPLE_VDIF, 'vdif')))
def test_open(sample, fmt):
    # Use extra arguments that allow all formats to succeed.
    extra_args = {'nchan': 8,
                  'ref_time': Time('2014-01-01'),
                  'sample_rate': 32.*u.MHz}
    info = file_info(sample, fmt, **extra_args)
    with baseband_open(sample, 'rs', **extra_args) as fh:
        assert fh.start_time == info.start_time


def test_open_write_checks():
    # Cannot have multiple formats for writing.
    with pytest.raises(ValueError):
        baseband_open('a.a', 'wb', fmt=('dada', 'mark4'))
