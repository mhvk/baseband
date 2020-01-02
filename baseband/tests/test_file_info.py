# Licensed under the GPLv3 - see LICENSE
import importlib
import pytest
from astropy.time import Time

from .. import file_info
from ..data import (SAMPLE_MARK4 as SAMPLE_M4, SAMPLE_MARK5B as SAMPLE_M5B,
                    SAMPLE_VDIF, SAMPLE_MWA_VDIF as SAMPLE_MWA, SAMPLE_DADA,
                    SAMPLE_PUPPI, SAMPLE_GSB_RAWDUMP_HEADER,
                    SAMPLE_GSB_RAWDUMP, SAMPLE_GSB_PHASED_HEADER,
                    SAMPLE_GSB_PHASED)


@pytest.mark.parametrize(
    ('sample', 'format_', 'missing', 'readable', 'error_keys'),
    ((SAMPLE_M4, 'mark4', True, True, []),
     (SAMPLE_M5B, 'mark5b', True, False, []),
     (SAMPLE_VDIF, 'vdif', False, True, []),
     (SAMPLE_MWA, 'vdif', False, True, ['frame_rate']),
     (SAMPLE_DADA, 'dada', False, True, []),
     (SAMPLE_PUPPI, 'guppi', False, True, []),
     (SAMPLE_GSB_RAWDUMP_HEADER, 'gsb', True, None, []),
     (SAMPLE_GSB_PHASED_HEADER, 'gsb', True, None, [])))
def test_basic_file_info(sample, format_, missing, readable, error_keys):
    info = file_info(sample)
    info_dict = info()
    assert info.format == format_
    assert info_dict['format'] == format_
    assert (hasattr(info, 'missing') and info.missing != {}) is missing
    assert ('missing' in info_dict and info_dict['missing'] != {}) is missing
    assert info.readable is readable
    assert list(info.errors.keys()) == error_keys


@pytest.mark.parametrize(
    ('sample', 'missing'),
    ((SAMPLE_M4, {'decade', 'ref_time'}),
     (SAMPLE_M5B, {'kday', 'ref_time', 'nchan'})))
def test_open_missing_args(sample, missing):
    info = file_info(sample)
    assert info.missing
    assert set(info.missing) == missing


@pytest.mark.parametrize(
    ('sample', 'format_', 'used', 'consistent', 'inconsistent'),
    ((SAMPLE_M4, 'mark4', ('ref_time',), ('nchan',), ()),
     (SAMPLE_M5B, 'mark5b', ('ref_time', 'nchan'), (), ()),
     (SAMPLE_VDIF, 'vdif', (), ('nchan', 'ref_time'), ()),
     (SAMPLE_DADA, 'dada', (), ('ref_time',), ('nchan',)),
     (SAMPLE_PUPPI, 'guppi', (), ('nchan',), ('ref_time',))))
def test_file_info(sample, format_, used, consistent, inconsistent):
    # Pass on extra arguments needed to get Mark4 and Mark5B to pass.
    # For GSB, we also need raw files, so we test that below.
    extra_args = {'ref_time': Time('2014-01-01'),
                  'nchan': 8}
    info = file_info(sample, **extra_args)
    assert info.format == format_
    info_dict = info()
    for attr in info.attr_names:
        info_value = getattr(info, attr)
        assert info_value is not None
        assert attr in info_dict or info_value == {}
    assert set(info.used_kwargs.keys()) == set(used)
    assert set(info.consistent_kwargs.keys()) == set(consistent)
    assert set(info.inconsistent_kwargs.keys()) == set(inconsistent)
    assert set(info.irrelevant_kwargs.keys()) == set()
    # Check that extraneous arguments get classified correctly.
    info2 = file_info(sample, life=42, **extra_args)
    assert info2.used_kwargs == info.used_kwargs
    assert info2.consistent_kwargs == info.consistent_kwargs
    assert info2.inconsistent_kwargs == info.inconsistent_kwargs
    assert info2.irrelevant_kwargs == {'life': 42}
    # Check we can indeed open a file with the extra arguments.
    module = importlib.import_module('.' + info.format, package='baseband')
    with module.open(sample, mode='rs', **info.used_kwargs) as fh:
        info3 = fh.info
    assert info3() == info_dict
    # Check that things properly do *not* work on a closed file.
    with module.open(sample, mode='rs', **info.used_kwargs) as fh:
        pass
    info4 = fh.info
    assert not info4
    assert 'File closed' in repr(info4)
    assert 'errors' in info4()
    assert any(isinstance(v, ValueError) for v in info4.errors.values())


@pytest.mark.parametrize(
    ('sample', 'raw', 'mode'),
    ((SAMPLE_GSB_RAWDUMP_HEADER, SAMPLE_GSB_RAWDUMP, 'rawdump'),
     (SAMPLE_GSB_PHASED_HEADER, SAMPLE_GSB_PHASED, 'phased')))
def test_gsb_with_raw_files(sample, raw, mode):
    # Note that the payload size in our sample files is reduced,
    # so without anything the file is not readable.
    bad_info = file_info(sample, raw=raw)
    assert bad_info.readable is False
    assert list(bad_info.errors.keys()) == ['frame0']
    # But with the correct sample_rate, it works.
    base_info = file_info(sample)
    sample_rate = 2**12 * base_info.frame_rate
    info = file_info(sample, raw=raw, sample_rate=sample_rate)
    assert info.format == 'gsb'
    assert info.readable is True
    assert not info.errors
    module = importlib.import_module('.' + info.format, package='baseband')
    # Check we can indeed open a file with the extra arguments.
    with module.open(sample, mode='rs', **info.used_kwargs) as fh:
        info2 = fh.info
    assert info2() == info()
