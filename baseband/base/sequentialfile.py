# Licensed under the GPLv3 - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import io
from bisect import bisect
import numpy as np


class SequentialFileBase(object):
    """Deal with several files as if they were one contiguous one."""
    def __init__(self, files, mode):
        self.files = files
        self.mode = mode
        self.offset = 0
        self.file_nr = None

    # Providing normal File IO properties.
    def readable(self):
        return self.fh.readable()

    def writable(self):
        return self.fh.writable()

    def seekable(self):
        return self.fh.readable()

    @property
    def closed(self):
        return self.fh.closed

    def open(self, file_nr):
        if file_nr != self.file_nr:
            if self.file_nr is not None:
                self.fh.close()
            try:
                fil = self.files[file_nr]
            except IndexError:
                raise OSError('ran out of files.')
            self.fh = io.open(fil, mode=self.mode)
            self.file_nr = file_nr

    def tell(self):
        return self.offset

    def close(self):
        self.fh.close()
        self.file_nr = None

    def memmap(self, dtype=np.uint8, mode=None, offset=None, shape=None,
               order='C'):
        """Map part of the file in memory.  Cannnot span file boundaries."""
        if self.closed:
            raise ValueError('memmap of closed file.')

        dtype = np.dtype(dtype)

        if mode is None:
            mode = self.mode.replace('b', '')

        if offset is not None and offset != self.offset:
            # move to correct file and offset (will fail for writable file).
            self.seek(offset)
        elif self.fh.tell() == self.file_size:
            self.open(self.file_nr + 1)

        if shape is None:
            count = self.size - self.offset
            if count % dtype.itemsize:
                raise ValueError("Size of available data is not a "
                                 "multiple of the data-type size.")
            shape = (count // dtype.itemsize,)
        else:
            if not isinstance(shape, tuple):
                shape = (shape,)
            count = dtype.itemsize
            for k in shape:
                count *= k

        if self.fh.tell() + count > self.file_size:
            raise ValueError('mmap length is greater than file size')

        mm = np.memmap(self.fh, dtype, mode, self.fh.tell(), shape, order)
        self.offset += count
        if 'r' in self.mode:
            self.seek(self.offset)
        return mm

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self):
        base_repr = ("{0}(files={1}, mode='{2}')"
                     .format(self.__class__.__name__, self.files, self.mode))
        extra = ("At offset: {0}; open file: {1!r}."
                 .format(self.tell(), None if self.file_nr is None else
                         self.files[self.file_nr]))
        return base_repr + "\n# " + extra


class SequentialFileReader(SequentialFileBase):
    """Read several files as if they were one contiguous one."""
    def __init__(self, files, mode='rb'):
        super(SequentialFileReader, self).__init__(files, mode)
        self.nfiles = len(files)
        self.file_sizes = []
        self.file_offsets = [0]
        size = 0
        for i in range(self.nfiles):
            self.open(i)
            file_size = self.fh.seek(0, 2)
            self.file_sizes.append(file_size)
            size += file_size
            self.file_offsets.append(size)
        self.size = size
        self.open(0)

    @property
    def file_size(self):
        return self.file_sizes[self.file_nr]

    def seek(self, offset, whence=0):
        if self.closed:
            raise ValueError('seek of closed file.')

        if whence == 1:
            offset += self.offset
        elif whence == 2:
            offset += self.size
        elif whence != 0:
            raise ValueError("invalid 'whence'; should be 0, 1, or 2.")

        if offset < 0:
            raise OSError('invalid offset')

        file_offset = offset - self.file_offsets[self.file_nr]
        if file_offset < 0 or file_offset >= self.file_sizes[self.file_nr]:
            file_nr = min(bisect(self.file_offsets, offset), self.nfiles) - 1
            self.open(file_nr)
            file_offset = offset - self.file_offsets[file_nr]
        self.fh.seek(file_offset)
        self.offset = offset

    def read(self, count=0):
        if self.closed:
            raise ValueError('read of closed file.')

        if count <= 0:
            count = self.size - self.tell()

        data = None
        while count > 0:
            extra = self.fh.read(count)
            if not extra:
                break
            if not data:  # avoid copies for first read.
                data = extra
            else:
                data += extra
            # Move offset pointer, possibly opening new file.
            self.seek(len(extra), 1)
            count -= len(extra)

        return data

    def memmap(self, dtype=np.uint8, mode=None, offset=None, shape=None,
               order='C'):
        """Map part of the file in memory.  Cannnot span file boundaries."""
        mm = super(SequentialFileReader, self).memmap(dtype, mode, offset,
                                                      shape, order)
        # Ensure file pointer is moved forward.
        self.seek(self.offset)
        return mm


class SequentialFileWriter(SequentialFileBase):
    """Write several files as if they were one contiguous one."""
    def __init__(self, files, mode='wb', file_size=None):
        super(SequentialFileWriter, self).__init__(files, mode)
        self.file_size = file_size
        self.file_offset = 0
        self.open(0)

    def seekable(self):
        return False

    def readable(self):
        return False

    def write(self, data):
        if self.closed:
            raise ValueError('write to closed file.')
        offset0 = self.offset
        if self.file_size is not None:
            remaining = self.file_size - self.fh.tell()
            while len(data) > remaining:
                self.fh.write(data[:remaining])
                self.offset += remaining
                data = data[remaining:]
                self.open(self.file_nr + 1)
                remaining = self.file_size

        self.fh.write(data)
        self.offset += len(data)
        return self.offset - offset0

    def memmap(self, dtype=np.uint8, mode=None, offset=None, shape=None,
               order='C'):
        """Map part of the file in memory.  Cannnot span file boundaries."""
        if shape is None:
            raise ValueError('cannot make writable memmap without shape.')
        return super(SequentialFileWriter, self).memmap(dtype, mode, offset,
                                                        shape, order)


def open(files, mode='rb', *args, **kwargs):
    if 'r' in mode and 'b' in mode:
        return SequentialFileReader(files, 'rb', *args, **kwargs)
    elif 'w' in mode and 'b' in mode:
        return SequentialFileWriter(files, 'w+b', *args, **kwargs)
    else:
        raise ValueError("invalid mode '{0}'".format(mode))
