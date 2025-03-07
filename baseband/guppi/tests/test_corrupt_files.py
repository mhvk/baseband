# Licensed under the GPLv3 - see LICENSE
import pytest
from numpy.testing import assert_array_equal

from ... import guppi
from ...data import SAMPLE_PUPPI as SAMPLE_FILE


class TestCorruptSampleCopy:
    def setup_class(cls):
        with open(SAMPLE_FILE, 'rb') as fh:
            cls.sample_bytes = fh.read()
        with guppi.open(SAMPLE_FILE, 'rs') as fs:
            cls.frame_nbytes = fs.header0.frame_nbytes
            cls.frame_rate = fs._frame_rate
            cls.start_time = fs.start_time
            cls.stop_time = fs.stop_time
            cls.data = fs.read()

    @pytest.mark.parametrize("remove", [
        1,  # last byte missing.
        16383,  # all but 1 byte
        16384,  # all frame data
        16384+1,  # also 1 header byte
        16384+6300,  # also most of header
    ])
    def test_missing_end(self, remove, tmpdir):
        filename = str(tmpdir.join('corrupted.m5b'))
        with open(filename, 'wb') as fw:
            fw.write(self.sample_bytes[:-remove])

        with guppi.open(filename) as fr:
            info = fr.info()
            data = fr.read()

        assert "warnings" in info
        warnings = info["warnings"]
        assert len(warnings) == 1
        assert "number_of_frames" in warnings
        assert "file contains non-integer" in warnings["number_of_frames"]
        assert info["shape"] == (2944,) + self.data.shape[1:]
        assert_array_equal(data, self.data[:2944])

    @pytest.mark.parametrize("extra", [
        1,  # extra byte
        6383,  # most of header worth
        7000,  # more than header
        16384+6400+10,  # full frame worth and a bit
    ])
    def test_extra_junk(self, extra, tmpdir):
        filename = str(tmpdir.join('corrupted.m5b'))
        with open(filename, 'wb') as fw:
            fw.write(self.sample_bytes + self.sample_bytes[6400:6400+extra])

        with guppi.open(filename) as fr:
            info = fr.info()
            data = fr.read()

        assert "warnings" in info
        warnings = info["warnings"]
        if extra >= self.frame_nbytes:
            assert len(warnings) == 2
            assert "last_header" in warnings
            assert "unreadable and skipped" in warnings["last_header"]
        else:
            assert len(warnings) == 1
        assert "number_of_frames" in warnings
        assert "file contains non-integer" in warnings["number_of_frames"]
        assert info["shape"] == self.data.shape
        assert_array_equal(data, self.data)
