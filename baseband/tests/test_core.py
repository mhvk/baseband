# Licensed under the GPLv3 - see LICENSE
import pytest
import numpy as np
from astropy.time import Time
import astropy.units as u

from .. import open as baseband_open, file_info
from ..helpers import sequentialfile as sf
from ..data import (SAMPLE_MARK4 as SAMPLE_M4, SAMPLE_MARK5B as SAMPLE_M5B,
                    SAMPLE_VDIF, SAMPLE_DADA)


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


def test_open_squeeze():
    with baseband_open(SAMPLE_VDIF, 'rs', squeeze=False) as fh:
        assert fh.sample_shape == (8, 1)
    with baseband_open(SAMPLE_VDIF, 'rs', squeeze=True) as fh:
        assert fh.sample_shape == (8,)


@pytest.mark.parametrize('verify', [True, False, 'fix'])
def test_open_verify(verify):
    with baseband_open(SAMPLE_VDIF, 'rs', verify=verify) as fh:
        assert fh.verify == verify


def test_open_wrong_args():
    # Use extra arguments that allow formats to succeed up to the stream
    # reader, but then fail on an incorrect sample rate.
    mark4_args = {'nchan': 8,
                  'ref_time': Time('2014-01-01')}
    with pytest.raises(ValueError, match='inconsistent'):  # wrong sample_rate
        baseband_open(SAMPLE_M4, 'rs', sample_rate=31*u.MHz, **mark4_args)

    with pytest.raises(TypeError, match='unexpected'):  # extraneous argument
        baseband_open(SAMPLE_M4, 'rs', life=42, **mark4_args)

    with pytest.raises(ValueError, match='inconsistent'):  # wrong decade
        baseband_open(SAMPLE_VDIF, 'rs', decade=2000)

    with pytest.raises(ValueError, match='inconsistent'):  # wrong kday
        baseband_open(SAMPLE_VDIF, 'rs', kday=55000)

    with pytest.raises(ValueError, match='inconsistent'):  # ref_time off.
        baseband_open(SAMPLE_VDIF, 'rs', ref_time=Time('J2000'))

    with pytest.raises(ValueError, match='inconsistent'):  # nchan wrong.
        baseband_open(SAMPLE_DADA, 'rs', nchan=8)

    with pytest.raises(TypeError):  # decade not int
        baseband_open(SAMPLE_M4, 'rs', decade='2010')

    with pytest.raises(TypeError):  # kday not int
        baseband_open(SAMPLE_M5B, 'rs', kday='unknown', nchan=8, bps=2)


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
    with pytest.raises(ValueError, match='cannot specify multiple'):
        baseband_open('a.a', 'wb', fmt=('dada', 'mark4'))


def test_unsupported_file(tmpdir):
    name = str(tmpdir.join('test.unsupported'))
    with open(name, 'wb') as fw:
        fw.write(b'abcdefghijklmnopqrstuvwxyz')

    with pytest.raises(ValueError, match='could not be auto-determined'):
        baseband_open(name)


def test_format_with_no_info(monkeypatch):
    monkeypatch.delattr('baseband.vdif.info')
    info = file_info(SAMPLE_VDIF)
    assert info
    assert info.format == 'vdif'
    assert not hasattr(info, 'used_kwargs')

    with baseband_open(SAMPLE_VDIF, format=('vdif', 'mark5b')) as fh:
        assert fh.info.format == 'vdif'

    with pytest.raises(ValueError, match='could not be auto-determined'):
        baseband_open(SAMPLE_M4, format=('vdif', 'mark5b'))
