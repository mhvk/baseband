# Licensed under the GPLv3 - see LICENSE
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pytest
import numpy as np
from astropy.time import Time
import astropy.units as u

from .. import open as baseband_open, file_info
from ..helpers import sequentialfile as sf
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


@pytest.mark.parametrize('sample', (SAMPLE_M4, SAMPLE_M5B))
def test_open_missing_args(sample):
    with pytest.raises(TypeError) as exc:
        baseband_open(sample, 'rs')
    assert "missing required arguments" in str(exc.value)


def test_open_wrong_args():
    # Use extra arguments that allow formats to succeed up to the stream
    # reader, but then fail on an incorrect sample rate.
    mark4_args = {'nchan': 8,
                  'ref_time': Time('2014-01-01')}
    with pytest.raises(ValueError) as exc:  # wrong sample_rate
        baseband_open(SAMPLE_M4, 'rs', sample_rate=31*u.MHz, **mark4_args)
    assert "inconsistent" in str(exc.value)

    with pytest.raises(TypeError) as exc:  # extraneous argument
        baseband_open(SAMPLE_M4, 'rs', life=42, **mark4_args)
    assert "unexpected" in str(exc.value)

    with pytest.raises(ValueError) as exc:  # wrong decade
        baseband_open(SAMPLE_VDIF, 'rs', decade=2000)
    assert "inconsistent" in str(exc.value)

    with pytest.raises(ValueError) as exc:  # wrong kday
        baseband_open(SAMPLE_VDIF, 'rs', kday=55000)
    assert "inconsistent" in str(exc.value)

    with pytest.raises(ValueError) as exc:  # ref_time too far off.
        baseband_open(SAMPLE_VDIF, 'rs', ref_time=Time('J2000'))
    assert "inconsistent" in str(exc.value)

    with pytest.raises(ValueError) as exc:  # inconsistent nchan.
        baseband_open(SAMPLE_DADA, 'rs', nchan=8)
    assert "inconsistent" in str(exc.value)


def test_open_sequence(tmpdir):
    """Test opening file sequences with `baseband.open`."""
    # Open DADA file sequence by passing a list.
    with baseband_open(SAMPLE_DADA) as fh:
        data1 = fh.read()
        header1 = fh.header0.copy()
    header1.payload_nbytes = header1.payload_nbytes // 2

    files = [str(tmpdir.join('f.{0:d}.dada'.format(x))) for x in range(2)]
    with baseband_open(files, 'ws', format='dada', header0=header1) as fw:
        fw.write(data1)

    with baseband_open(files) as fn:
        assert fn.info.format == 'dada'
        assert len(fn.fh_raw.files) == 2
        assert fn.header0 == header1
        assert np.all(fn.read() == data1)

    # Open VDIF file sequence by passing a FileNameSequencer.
    files = sf.FileNameSequencer(str(tmpdir.join('f{file_nr:03d}.vdif')))
    with baseband_open(SAMPLE_VDIF) as fh:
        data2 = fh.read()
        header2 = fh.header0.copy()

    # Sample file has 2 framesets of 8 frames each.
    with baseband_open(files, 'ws', format='vdif', nthread=8,
                       file_size=8*header2.frame_nbytes, **header2) as fw:
        fw.write(data2)

    # Can't check header0 because frameset might be out of order.
    with baseband_open(files) as fn:
        assert fn.info.format == 'vdif'
        assert len(fn.fh_raw.files) == 2
        assert np.all(data2 == fn.read())


def test_open_write_checks():
    # Cannot have multiple formats for writing.
    with pytest.raises(ValueError):
        baseband_open('a.a', 'wb', fmt=('dada', 'mark4'))
