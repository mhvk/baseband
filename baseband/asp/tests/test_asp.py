import pytest

from ... import asp
from ...data import SAMPLE_ASP as SAMPLE_FILE


__all__ = ['TestASP']


class TestASP:
    def setup_class(cls):
        with open(SAMPLE_FILE, 'rb') as fh:
            cls.file_header = asp.ASPFileHeader.fromfile(fh)
            cls.block_header = asp.ASPHeader.fromfile(fh)
            cls.header = asp.ASPHeader(cls.block_header.words, cls.file_header)
            cls.payload = asp.ASPPayload.fromfile(
                fh, header=cls.header)
            cls.tell_at_end = fh.tell()

    def test_setup(self):
        assert self.tell_at_end == (
            self.header.nbytes
            + self.file_header.nbytes
            + self.payload.nbytes), 'ended in wrong spot'

    def test_block_header(self):
        header = self.block_header
        assert header.nbytes == 44
        assert header['totalsize'] == 512
        assert header['nptssend'] == 128
        assert header['freqchanno'] == 10
        with pytest.raises(KeyError):
            header['n_chan']

    def test_file_header(self):
        header = self.file_header
        assert header['n_chan'] == 8

    def test_header(self):
        header = self.header
        assert header.nbytes == 44  # block header only!
        # Information from both block and file headers.
        assert header['totalsize'] == 512
        assert header['nptssend'] == 128
        assert header['freqchanno'] == 10
        assert header['n_chan'] == 8

    def test_file_streamer(self):
        with asp.open(SAMPLE_FILE, 'rs'):
            pass
