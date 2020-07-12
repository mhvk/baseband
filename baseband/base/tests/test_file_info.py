# Licensed under the GPLv3 - see LICENSE
"""Test things that are not already tested with the many formats."""
from ... import vdif, data


def test_stream_info_without_file_info(monkeypatch):
    """Test that stream info is possible without file info.

    This is the case for scintillometry.io.hdf5.  We use a hack where
    we replace the regular VDIF file reader with an info-less one.
    """
    class NoFileInfoReader(vdif.base.VDIFFileReader):
        @property
        def info(self):
            raise AttributeError

    monkeypatch.setattr('baseband.vdif.base.VDIFFileReader',
                        NoFileInfoReader)
    with vdif.open(data.SAMPLE_VDIF) as fh:
        info = fh.info
        assert info.file_info is None
        assert info.format == 'vdif'
