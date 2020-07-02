import numpy as np
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
                fh, header=cls.block_header)
            cls.frame = asp.ASPFrame(cls.header, cls.payload)
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

    def test_payload(self):
        payload = self.payload
        assert payload.sample_shape == (2,)
        assert payload.shape == (self.header.samples_per_frame, 2)
        assert payload.complex_data
        assert payload.bps == 8
        assert payload.dtype == np.dtype('c8')

    def test_frame(self):
        frame = self.frame
        assert frame.sample_shape == (2,)
        assert frame.shape == (self.header.samples_per_frame, 2)
        assert frame.dtype == np.dtype('c8')
        assert frame['freqchanno'] == 10
        assert frame['n_chan'] == 8

    def test_file_reader_read_header(self):
        with asp.open(SAMPLE_FILE, 'rb') as fh:
            file_header = fh.read_file_header()
            block_header = fh.read_header()
            fh.seek(-block_header.nbytes, 1)
            header = fh.read_header(file_header=file_header)

        assert file_header == self.file_header
        assert block_header == self.block_header
        assert block_header != self.header
        assert header == self.header
        assert header != self.block_header

    def test_file_reader_full_header_frame(self):
        with asp.open(SAMPLE_FILE, 'rb') as fh:
            file_header = fh.read_file_header()
            frame = fh.read_frame(file_header=file_header)

        assert file_header == self.file_header
        assert frame == self.frame
        assert frame.header == self.header
        assert frame.payload == self.payload

    def test_file_streamer(self):
        with asp.open(SAMPLE_FILE, 'rs') as fh:
            header0 = fh.header0

        assert header0 == self.header
