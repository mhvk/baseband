# Licensed under the GPLv3 - see LICENSE
"""Test things that are not already tested with the many formats."""
import pytest

from ... import vdif, data
from ..file_info import FileReaderInfo, StreamReaderInfo


def test_str_repr():
    # Check names are interpreted correctly for all types of info_item
    # Directly assigned
    assert str(FileReaderInfo.errors).startswith('errors: ')
    # From parent
    assert str(StreamReaderInfo.start_time).startswith('start_time: ')
    # From header0
    assert str(FileReaderInfo.bps).startswith('bps: ')
    # From function
    assert str(FileReaderInfo.header0).startswith('header0: ')
    # From function with needs (i.e., via __call__)
    assert str(FileReaderInfo.readable).startswith('readable: ')
    assert repr(FileReaderInfo.errors).startswith('<info_item errors')
    assert repr(FileReaderInfo()).startswith('FileReaderInfo (unbound)')
    assert repr(StreamReaderInfo()).startswith('StreamReaderInfo (unbound)')
    assert repr(vdif.base.VDIFFileReader.info).startswith(
        'VDIFFileReaderInfo (unbound)')

    with pytest.raises(TypeError, match="assigned 'info_item'"):
        StreamReaderInfo.continuous('a')


def test_stream_info_without_file_info(monkeypatch):
    """Test that stream info is possible without file info.

    This is the case for scintillometry.io.hdf5.  We use a hack where
    we replace the regular VDIF file reader with an info-less one.
    """
    class NoFileInfoReader(vdif.base.VDIFFileReader):
        @property
        def info(self):
            raise AttributeError

    monkeypatch.setattr(vdif.base, 'VDIFFileReader', NoFileInfoReader)
    with vdif.open(data.SAMPLE_VDIF) as fh:
        info = fh.info
        assert info.file_info is None
        assert info.format == 'vdif'
