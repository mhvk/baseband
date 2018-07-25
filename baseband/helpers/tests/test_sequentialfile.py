# Licensed under the GPLv3 - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import numpy as np
from astropy.tests.helper import pytest

from .. import sequentialfile as sf

from baseband import vdif
from baseband.data import SAMPLE_VDIF as SAMPLE_FILE


class Sequencer(object):
    def __init__(self, template):
        self.template = template

    def __getitem__(self, item):
        return self.template.format(item)


class TestSequentialFileReader(object):

    def _setup(self, tmpdir):
        self.data = b'abcdefghijklmnopqrstuvwxyz'
        self.uint8_data = np.frombuffer(self.data, dtype=np.uint8)
        self.size = len(self.data)
        self.files = [str(tmpdir.join('file{:1d}.raw'.format(i)))
                      for i in range(3)]
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

    def test_setup(self, tmpdir):
        self._setup(tmpdir)
        assert self.sizes == [10, 10, 6]
        assert self.offsets == [0, 10, 20, 26]

    def test_open_close(self, tmpdir):
        self._setup(tmpdir)

        fh = sf.open(self.files)
        assert fh.readable()
        assert fh.seekable()
        assert not fh.writable()
        assert not fh.closed
        assert repr(fh).startswith('SequentialFileReader(files=')
        fh.close()
        assert fh.closed
        with pytest.raises(ValueError):
            # wrong mode: no r or w
            sf.open(self.files, 'b')
        with pytest.raises(TypeError):
            # cannot pass in file_size for reading
            sf.open(self.files, 'rb', file_size=10)

    def test_context(self, tmpdir):
        self._setup(tmpdir)
        with sf.open(self.files) as fh:
            assert fh.readable()
            assert fh.seekable()
            assert not fh.writable()
            assert not fh.closed
        assert fh.closed

    def test_attributes(self, tmpdir):
        self._setup(tmpdir)
        with sf.open(self.files) as fh:
            assert fh.files == self.files
            assert fh.tell() == 0
            assert fh.size == self.size
            assert fh._file_sizes == self.sizes
            assert fh._file_offsets == self.offsets
            with pytest.raises(AttributeError):
                fh.bla

    def test_seek(self, tmpdir):
        self._setup(tmpdir)

        with sf.open(self.files) as fh:
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
            fh.seek(10, 2)
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
        # cannot seek closed file
        with pytest.raises(ValueError):
            fh.seek(10)

    def test_read(self, tmpdir):
        self._setup(tmpdir)

        with sf.open(self.files) as fh:
            check = fh.read(2)
            assert check == self.data[:2]
            check = fh.read(2)
            assert check == self.data[2:4]
            check = fh.read(10)
            assert check == self.data[4:14]
            check = fh.read()
            assert check == self.data[14:]
            fh.seek(0)
        # cannot read closed file
        with pytest.raises(ValueError):
            fh.read()

    def test_read_all(self, tmpdir):
        self._setup(tmpdir)
        with sf.open(self.files) as fh:
            check = fh.read()
            assert check == self.data

    def test_seek_read(self, tmpdir):
        self._setup(tmpdir)
        with sf.open(self.files) as fh:
            fh.seek(8)
            assert fh.read(8) == self.data[8:16]
            fh.seek(1, 1)
            assert fh.read(8) == self.data[17:25]
            fh.seek(-2, 1)
            assert fh.read(2) == self.data[23:25]
            fh.seek(-10, 2)
            assert fh.read() == self.data[-10:]
            fh.seek(10, 2)
            assert fh.read() == b''
            assert fh.read(10) == b''

    def test_memmap(self, tmpdir):
        self._setup(tmpdir)

        with sf.open(self.files) as fh:
            mm = fh.memmap(offset=0, shape=(5,))
            assert fh.tell() == 5
            assert (mm == self.uint8_data[:5]).all()
            mm = fh.memmap(shape=(5,))
            assert fh.tell() == 10
            assert (mm == self.uint8_data[5:10]).all()
            with pytest.raises(ValueError):
                fh.memmap(offset=7, shape=(5,))
            offset = self.offsets[1]
            fh.seek(offset)
            mm = fh.memmap(shape=5)
            assert (mm == self.uint8_data[offset:offset+5]).all()
            fh.seek(-2, 2)
            mm = fh.memmap()
            assert (mm == self.uint8_data[-2:]).all()
            fh.seek(-4, 2)
            mm = fh.memmap(mode='r', dtype=np.uint16)
            assert (mm == self.uint8_data[-4:].view(np.uint16)).all()
            fh.seek(-3, 2)
            with pytest.raises(ValueError):
                fh.memmap(dtype=np.uint16)

        with pytest.raises(ValueError):  # file closed.
            fh.memmap(offset=0, shape=(5,))


class TestSequentialFileWriter(object):
    def _setup(self, tmpdir):
        self.data = b'abcdefghijklmnopqrstuvwxyz'
        self.uint8_data = np.frombuffer(self.data, dtype=np.uint8)
        self.files = [str(tmpdir.join('file{:1d}.raw'.format(i)))
                      for i in range(3)]

    def files_exist(self):
        return [os.path.isfile(fil) for fil in self.files]

    def test_open_close(self, tmpdir):
        self._setup(tmpdir)
        fh = sf.open(self.files, 'wb', file_size=10)
        assert fh.writable()
        assert not fh.closed
        fh.close()
        assert fh.closed
        assert self.files_exist() == [True, False, False]

    def test_context(self, tmpdir):
        self._setup(tmpdir)
        with sf.open(self.files, 'wb', file_size=10) as fh:
            assert fh.writable()
            assert not fh.closed

        assert fh.closed
        assert self.files_exist() == [True, False, False]

    def test_attributes(self, tmpdir):
        self._setup(tmpdir)
        with sf.open(self.files, 'wb', file_size=10) as fh:
            assert fh.files == self.files
            assert fh.file_size == 10
            assert fh.tell() == 0

    def test_write(self, tmpdir):
        self._setup(tmpdir)
        with sf.open(self.files, 'wb', file_size=10) as fh:
            fh.write(self.data[:5])
            assert fh.tell() == 5
            assert fh.file_nr == 0
            assert self.files_exist() == [True, False, False]
            fh.write(self.data[5:20])
            assert fh.tell() == 20
            assert fh.file_nr == 1
            assert self.files_exist() == [True, True, False]
            fh.write(self.data[20:])
            assert fh.tell() == len(self.data)
            assert fh.file_nr == 2

        # Cannot write to closed file.
        with pytest.raises(ValueError):
            fh.write(b' ')

        assert self.files_exist() == [True, True, True]
        with sf.open(self.files, 'rb') as fh:
            assert fh.read() == self.data

        with pytest.raises(OSError):
            with sf.open(self.files, 'wb', file_size=10) as fh:
                fh.write(b' ' * (len(self.files) * 10 + 1))

    def test_write_all(self, tmpdir):
        self._setup(tmpdir)
        with sf.open(self.files, 'wb', file_size=10) as fh:
            fh.write(self.data)
            assert fh.tell() == len(self.data)
            assert fh.file_nr == 2

        assert self.files_exist() == [True, True, True]
        with sf.open(self.files, 'rb') as fh:
            assert fh.read() == self.data

    def test_simple_sequencer(self, tmpdir):
        self._setup(tmpdir)

        sequencer = Sequencer(str(tmpdir.join('file{:03d}.raw')))
        with sf.open(sequencer, 'wb', file_size=8) as fh:
            fh.write(self.data)
            assert fh.file_nr == 3

        with sf.open(sequencer, 'rb') as fh:
            check = fh.read()
            assert check == self.data
            assert fh._file_sizes == [8, 8, 8, 2]

    def test_memmap(self, tmpdir):
        self._setup(tmpdir)
        data = self.uint8_data
        with sf.open(self.files, 'w+b', file_size=10) as fh:
            mm = fh.memmap(shape=(5,))
            assert fh.tell() == 5
            mm[:] = data[:5]
            mm2 = fh.memmap(offset=5, shape=(5,))
            mm2[:4] = data[5:9]
            mm2[-1] = data[-1]
            mm3 = fh.memmap(shape=(5,))
            mm3[:] = data[10:15]
            with pytest.raises(ValueError):
                fh.memmap(shape=(6,))  # beyond file.
            with pytest.raises(ValueError):
                fh.memmap()  # Need to pass in shape
            with pytest.raises(ValueError):
                fh.memmap(offset=4)  # Cannot seek
            mm4 = fh.memmap(dtype=np.uint16, shape=(2,))
            mm4[:] = data[15:19].view(np.uint16)

        assert self.files_exist() == [True, True, False]
        with sf.open(self.files[:2], 'rb') as fh:
            check = fh.read()
        assert len(check) == 19
        assert check[:9] == self.data[:9]
        assert check[9] == self.data[-1]
        assert check[10:] == self.data[10:19]

    def test_memmap_fail(self, tmpdir):
        self._setup(tmpdir)
        with sf.open(self.files, 'wb', file_size=10) as fh:
            with pytest.raises(ValueError):
                fh.memmap(shape=(5,))


class TestFileNameSequencer(object):
    def setup(self):
        with open(SAMPLE_FILE, 'rb') as fh:
            self.header = vdif.VDIFHeader.fromfile(fh)

    def test_enumeration(self):
        fns1 = sf.FileNameSequencer('x{file_nr:03d}.vdif')
        assert fns1[0] == 'x000.vdif'
        assert fns1[107] == 'x107.vdif'
        fns2 = sf.FileNameSequencer('{SNAKE}_{file_nr}', {'SNAKE': 'python'})
        assert fns2[13] == 'python_13'

        with pytest.raises(KeyError):
            sf.FileNameSequencer('{snake:06d}.x', {'SNAKE': 10})

    def test_header_extraction(self):
        template = 'x.edv{edv}.stn_{station_id}.{file_nr:05d}.vdif'
        fns = sf.FileNameSequencer(template, self.header)
        assert fns[0] == 'x.edv3.stn_65532.00000.vdif'
        assert fns[133] == 'x.edv3.stn_65532.00133.vdif'

    def test_len(self, tmpdir):
        template = str(tmpdir.join('a{file_nr}.bin'))
        fns = sf.FileNameSequencer(template)
        for i in range(5):
            assert len(fns) == i
            filename = fns[i]
            assert filename.endswith('a{}.bin'.format(i))
            with open(filename, 'wb') as fh:
                fh.write(b'bird')
        assert len(fns) == 5
        assert fns[-2] == fns[3]
        assert fns[-1].endswith('a4.bin')
        with pytest.raises(IndexError):
            fns[-10]
