# Licensed under the GPLv3 - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from astropy.tests.helper import pytest

from .. import multifile as mf


class TestMultiFileReader(object):
    def setup(self):
        self.data = b'abcdefghijklmnopqrstuvwxyz'
        self.uint8_data = np.fromstring(self.data, dtype=np.uint8)
        self.size = len(self.data)
        self.files = ['file{:1d}.raw'.format(i) for i in range(3)]
        self.max_file_size = 10
        self.sizes = []
        self.offsets = [0]
        offset = 0
        for filename in self.files:
            with open(filename, 'wb') as fw:
                part = self.data[offset:offset+self.max_file_size]
                fw.write(part)
                self.sizes.append(len(part))
                self.offsets.append(self.offsets[-1] + len(part))
            offset += self.max_file_size

    def test_setup(self):
        assert self.sizes == [10, 10, 6]
        assert self.offsets == [0, 10, 20, 26]

    def test_open_close(self):
        fh = mf.open(self.files)
        assert fh.readable
        assert fh.seekable
        assert not fh.closed
        fh.close()
        assert fh.closed

    def test_context(self):
        with mf.open(self.files) as fh:
            assert fh.readable
            assert fh.seekable
            assert not fh.closed
        assert fh.closed

    def test_attributes(self):
        with mf.open(self.files) as fh:
            assert fh.files == self.files
            assert fh.nfiles == len(self.files)
            assert fh.file_sizes == self.sizes
            assert fh.file_offsets == self.offsets
            assert fh.size == self.size

    def test_seek(self):
        with mf.open(self.files) as fh:
            fh.seek(self.offsets[1])
            assert fh.file_nr == 1
            assert fh.tell() == self.offsets[1]
            fh.seek(self.offsets[1] + 1)
            assert fh.file_nr == 1
            fh.seek(self.offsets[2])
            assert fh.file_nr == 2
            fh.seek(self.offsets[3])
            assert fh.tell() == self.size
            assert fh.file_nr == 2
            fh.seek(self.size)
            assert fh.file_nr == 2
            fh.seek(self.size + 10)
            assert fh.file_nr == 2
            assert fh.tell() == self.size + 10
            with pytest.raises(OSError):
                fh.seek(-3)
            fh.seek(-3, 2)
            assert fh.tell() == self.size - 3
            assert fh.file_nr == 2
            fh.seek(-5, 1)
            assert fh.tell() == self.size - 3 - 5
            assert fh.file_nr == 1
            with pytest.raises(ValueError):
                fh.seek(-5, -1)
            with pytest.raises(ValueError):
                fh.seek(-5, 3)

        def test_read(self):
            with mf.open(self.files) as fh:
                check = fh.read(2)
                assert check == self.data[:2]
                check = fh.read(2)
                assert check == self.data[2:4]
                check = fh.read(10)
                assert check == self.data[4:14]
                check = fh.read()
                assert check == self.data[14:]

    def test_read_all(self):
        with mf.open(self.files) as fh:
            check = fh.read()
            assert check == self.data

    def test_seek_read(self):
        with mf.open(self.files) as fh:
            fh.seek(8)
            assert fh.read(8) == self.data[8:16]
            fh.seek(1, 1)
            assert fh.read(8) == self.data[17:25]
            fh.seek(-2, 1)
            assert fh.read(2) == self.data[23:25]
            fh.seek(-10, 2)
            assert fh.read() == self.data[-10:]

    def test_memmap(self):
        with mf.open(self.files) as fh:
            mm = fh.memmap(mode='r', offset=0, shape=(5,))
            assert (mm == self.uint8_data[:5]).all()
            mm = fh.memmap(mode='r', offset=5, shape=(5,))
            assert (mm == self.uint8_data[5:10]).all()
            with pytest.raises(ValueError):
                fh.memmap(mode='r', offset=7, shape=(5,))
            offset = self.offsets[1]
            fh.seek(offset)
            mm = fh.memmap(mode='r', shape=(5,))
            assert (mm == self.uint8_data[offset:offset+5]).all()
            fh.seek(-2, 2)
            mm = fh.memmap(mode='r')
            assert (mm == self.uint8_data[-2:]).all()
            fh.seek(-4, 2)
            mm = fh.memmap(mode='r', dtype=np.uint16)
            assert (mm == self.uint8_data[-4:].view(np.uint16)).all()
            fh.seek(-3, 2)
            with pytest.raises(ValueError):
                fh.memmap(mode='r', dtype=np.uint16)
