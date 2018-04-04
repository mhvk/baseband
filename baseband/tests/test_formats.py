# Licensed under the GPLv3 - see LICENSE
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pytest
from astropy.time import Time
import astropy.units as u

from ..formats import file_info
from ..data import (SAMPLE_MARK4 as SAMPLE_M4, SAMPLE_MARK5B as SAMPLE_M5B,
                    SAMPLE_VDIF, SAMPLE_MWA_VDIF as SAMPLE_MWA, SAMPLE_DADA,
                    SAMPLE_GSB_RAWDUMP_HEADER, SAMPLE_GSB_PHASED_HEADER)


@pytest.mark.parametrize(
    ('sample', 'fmt', 'subfmt'),
    ((SAMPLE_M4, 'mark4', None),
     (SAMPLE_M5B, 'mark5b', None),
     (SAMPLE_VDIF, 'vdif', 'edv=3'),
     (SAMPLE_MWA, 'vdif', 'edv=0'),
     (SAMPLE_DADA, 'dada', None),
     (SAMPLE_GSB_RAWDUMP_HEADER, 'gsb', 'rawdump'),
     (SAMPLE_GSB_PHASED_HEADER, 'gsb', 'phased')))
def test_file_info(sample, fmt, subfmt):
    info = file_info(sample)
    assert info['fmt'] == fmt
    if subfmt is None:
        assert 'subfmt' not in info
    else:
        assert info['subfmt'] == subfmt
    if fmt.startswith('mark'):
        assert 'missing' in info
    else:
        assert 'missing' not in info


@pytest.mark.parametrize(
    ('sample', 'missing'),
    ((SAMPLE_M4, {'decade', 'ref_time'}),
     (SAMPLE_M5B, {'kday', 'ref_time', 'nchan'})))
def test_open_missing_args(sample, missing):
    info = file_info(sample)
    assert 'missing' in info
    assert set(info['missing']) == missing
