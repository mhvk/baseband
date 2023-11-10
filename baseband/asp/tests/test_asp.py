import numpy as np
from numpy.testing import assert_array_equal
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

        assert header.mutable is False
        with pytest.raises(TypeError):
            header['totalsize'] = 256

        header2 = header.copy()
        header2['totalsize'] = 256
        assert header2['totalsize'] == 256
        assert header2 != header
        header2['totalsize'] = 512
        assert header2 == header

        pattern, mask = header.invariant_pattern()
        assert_array_equal(pattern, np.atleast_1d(header.words).view('<u4'))
        assert_array_equal(mask, np.array([-1, -1] + [0]*8 + [-1],
                                          dtype='<i4').view('<u4'))

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

    @pytest.mark.parametrize('file_writer', (False, True))
    def test_reproduce_frame(self, file_writer, tmpdir):
        check = str(tmpdir.join('check.asp'))
        if file_writer:
            with asp.open(check, 'wb') as fw:
                fw.write_file_header(self.file_header)
                fw.write_frame(self.frame)
        else:
            with open(check, 'wb') as fw:
                self.file_header.tofile(fw)
                self.frame.tofile(fw)

        with open(SAMPLE_FILE, 'rb') as fh:
            expected = fh.read(self.file_header.nbytes + self.frame.nbytes)

        with open(check, 'rb') as fh:
            data = fh.read()

        assert data == expected

    def test_locate_frames(self):
        with asp.open(SAMPLE_FILE, 'rb') as fh:
            # Maximum only needed because sample file has frame size smaller
            # than file_header size.
            locs0 = fh.locate_frames(self.header, maximum=5000)
            assert locs0 == [self.file_header.nbytes,
                             self.file_header.nbytes+self.frame.nbytes]
            # Also go a frame before first one, with normal search.
            fh.seek(self.file_header.nbytes - self.frame.nbytes)
            locs0a = fh.locate_frames(self.header)
            assert locs0a == [self.file_header.nbytes]
            file_nbytes = fh.seek(0, 2)
            locs1 = fh.locate_frames(self.header, forward=False)
            assert locs1 == [file_nbytes - self.frame.nbytes]

    def test_find_header(self):
        with asp.open(SAMPLE_FILE, 'rb') as fh:
            # Maximum only needed because sample file has frame size smaller
            # than file_header size.
            header0 = fh.find_header(self.header, maximum=5000)
            assert header0 == self.block_header
            assert fh.tell() == self.file_header.nbytes
            file_nbytes = fh.seek(0, 2)
            header1 = fh.find_header(self.header, forward=False)
            assert fh.tell() == file_nbytes - self.frame.nbytes
            assert header1 != header0
            for key in 'totalsize', 'nptssend', 'imjd', 'fmjd', 'freqchanno':
                assert header1[key] == header0[key]

    @pytest.mark.xfail(reason='Sample file broken?')
    def test_file_streamer(self):
        # TODO: fix frame index calculation!!
        # Or more likely the sample file!?
        with asp.open(SAMPLE_FILE, 'rs', verify=False) as fh:
            header0 = fh.header0
            assert fh.start_time == fh.header0.time
            data = fh.read(len(self.frame)*2)

        assert header0 == self.header
        assert data.dtype == fh.dtype
        assert_array_equal(data[:len(self.frame)], self.frame.data)
        # TODO: add check of actual content of data!
        assert data.shape == fh.shape

    def test_stream_writer(self, tmpdir):
        check = str(tmpdir.join('check.asp'))
        with asp.open(SAMPLE_FILE, 'rs', verify=False) as fh:
            header0 = fh.header0
            data = fh.read(len(self.frame)*2)

        with asp.open(check, 'ws', header0=header0) as fw:
            fw.write(data)

        with asp.open(check, 'rs') as fr:
            recovered = fr.read()

        assert_array_equal(recovered, data)

    @pytest.mark.xfail(reason='sample file wrong??')
    def test_reproduce_stream(self, tmpdir):
        check = str(tmpdir.join('check.asp'))
        with asp.open(SAMPLE_FILE, 'rs') as fh:
            header0 = fh.header0
            data = fh.read()

        with asp.open(check, 'ws', header0=header0) as fw:
            fw.write(data)

        with open(check, 'rb') as fr:
            recovered = fr.read()

        with open(SAMPLE_FILE, 'rb') as fh:
            expected = fh.read()

        assert recovered == expected
