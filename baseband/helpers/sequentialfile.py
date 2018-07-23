# Licensed under the GPLv3 - see LICENSE
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import io
import os
import re
import itertools
from bisect import bisect
import numpy as np
from astropy.utils import lazyproperty

__all__ = ['FileNameSequencer', 'SequentialFileReader', 'SequentialFileWriter',
           'open']


class FileNameSequencer(object):
    """List-like generator of filenames using a template.

    The template is formatted, filling in any items in curly brackets with
    values from the header.  It is additionally possible to insert a file
    number equal to the indexing value, indicated with '{file_nr}'.

    The length of the instance will be the number of files that exist that
    match the template for increasing values of the file number (when writing,
    it is the number of files that have so far been generated).

    Parameters
    ----------
    template : str
        Template to format to get specific filenames.  Curly bracket item
        keywords are case-sensitive (eg. '{FRAME_NR}' or '{Frame_NR}' will not
        use ``header['frame_nr']``.
    header : dict-like
        Structure holding key'd values that are used to fill in the format.

    Examples
    --------

    >>> from baseband import vdif
    >>> from baseband.helpers import sequentialfile as sf
    >>> vfs = sf.FileNameSequencer('a{file_nr:03d}.vdif')
    >>> vfs[10]
    'a010.vdif'
    >>> from baseband.data import SAMPLE_VDIF
    >>> with vdif.open(SAMPLE_VDIF, 'rb') as fh:
    ...     header = vdif.VDIFHeader.fromfile(fh)
    >>> vfs = sf.FileNameSequencer('obs.edv{edv:d}.{file_nr:05d}.vdif', header)
    >>> vfs[10]
    'obs.edv3.00010.vdif'
    """
    def __init__(self, template, header={}):
        self.items = {}

        def check_and_convert(x):
            string = x.group()
            key = string[1:-1]
            if key != 'file_nr':
                self.items[key] = header[key]
            return string

        self.template = re.sub(r'{\w+[}:]', check_and_convert, template)

    def _process_items(self, file_nr):
        # No check for whether file_nr > len(self), since there may not be a
        # predeterminable length when writing.
        if file_nr < 0:
            file_nr += len(self)
            if file_nr < 0:
                raise IndexError('file number out of range.')

        self.items['file_nr'] = file_nr

    def __getitem__(self, file_nr):
        self._process_items(file_nr)
        return self.template.format(**self.items)

    def __len__(self):
        file_nr = 0
        while os.path.isfile(self[file_nr]):
            file_nr += 1

        return file_nr


class SequentialFileBase(object):
    """Deal with several files as if they were one contiguous one.

    For details, see `SequentialFileReader` and `SequentialFileWriter`.
    """
    def __init__(self, files, mode='rb', opener=None):
        self.files = files
        self.mode = mode
        self.opener = io.open if opener is None else opener
        self.file_nr = None
        self._file_sizes = []
        self._file_offsets = [0]
        self._open(0)

    def __getattr__(self, attr):
        """Try to get things on the current open file if it is not on self."""
        if not attr.startswith('_'):
            try:
                return getattr(self.fh, attr)
            except AttributeError:
                pass
        return self.__getattribute__(attr)

    def _open(self, file_nr):
        """Open the ``file_nr``th file of the list of underlying files.

        If a different file was already open, it is closed.  Nothing is done
        if the requested file is already open.
        """
        if file_nr != self.file_nr:
            try:
                fh = self.opener(self.files[file_nr], mode=self.mode)
            except IndexError:
                raise OSError('ran out of files.')
            if self.file_nr is not None:
                self.fh.close()
            self.fh = fh
            self.file_nr = file_nr
            if self.file_nr == len(self._file_sizes):
                file_size = self.file_size
                if file_size is not None:  # can happen for single-file write.
                    self._file_sizes.append(file_size)
                    self._file_offsets.append(self._file_offsets[-1] +
                                              file_size)

    def tell(self):
        """Return the current stream position."""
        return self._file_offsets[self.file_nr] + self.fh.tell()

    def memmap(self, dtype=np.uint8, mode=None, offset=None, shape=None,
               order='C'):
        """Map part of the file in memory.

        Note that the map cannnot span multiple underlying files.
        Parameters are as for `~numpy.memmap`.
        """
        if self.closed:
            raise ValueError('memmap of closed file.')

        dtype = np.dtype(dtype)

        if mode is None:
            mode = self.mode.replace('b', '')

        if offset is not None and offset != self.tell():
            # seek will fail for SequentialFileWriter, so we try to avoid it.
            self.seek(offset)
        elif self.fh.tell() == self._file_sizes[self.file_nr]:
            self._open(self.file_nr + 1)

        if shape is None:
            count = self.size - self.tell()
            if count % dtype.itemsize:
                raise ValueError("size of available data is not a "
                                 "multiple of the data-type size.")
            shape = (count // dtype.itemsize,)
        else:
            if not isinstance(shape, tuple):
                shape = (shape,)
            count = dtype.itemsize
            for k in shape:
                count *= k

        if self.fh.tell() + count > self._file_sizes[self.file_nr]:
            raise ValueError('mmap length exceeds individual file size')

        file_offset = self.fh.tell()
        mm = np.memmap(self.fh, dtype, mode, file_offset, shape, order)
        self.fh.seek(file_offset + count)
        return mm

    def close(self):
        """Close the currently open local file, and therewith the set."""
        if self.file_nr is not None:
            self.fh.close()
            self.file_nr = None

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
    """Read several files as if they were one contiguous one.

    Parameters
    ----------
    files : list, tuple, or other iterable of str, filehandle
        The contains the names of the underlying files that should be combined.
        If not a list or tuple, it should allow indexing with positive indices,
        and raise `IndexError` if these are out of range.
    mode : str, optional
        The mode with which the files should be opened (default: 'rb')
    opener : callable, optional
        Function to open a single file (default: `io.open`).
    """

    @property
    def file_size(self):
        """Size of the underlying file currently open for reading."""
        offset = self.fh.tell()
        file_size = self.fh.seek(0, 2)
        self.fh.seek(offset)
        return file_size

    @lazyproperty
    def size(self):
        """Size of all underlying files combined."""
        offset = self.tell()
        for i in itertools.count(start=len(self._file_sizes)):
            try:
                self._open(i)
            except Exception:
                break

        self.seek(offset)
        return self._file_offsets[-1]

    def seek(self, offset, whence=0):
        if self.closed:
            raise ValueError('seek of closed file.')

        if whence == 1:
            offset += self.tell()
        elif whence == 2:
            offset += self.size
        elif whence != 0:
            raise ValueError("invalid 'whence'; should be 0, 1, or 2.")

        if offset < 0:
            raise OSError('invalid offset')

        # If the offset is not in the current file, find right one.
        while not (0 <= offset - self._file_offsets[self.file_nr] <
                   self._file_sizes[self.file_nr]):
            # Note that not all files may have been opened at this point.
            # In that case, bisecting would find we're out of the current files
            # and one would open a new one.  The while loop ensures we keep
            # trying until we've got there or reached the end of the files.
            file_nr = bisect(self._file_offsets, offset) - 1
            try:
                self._open(file_nr)
            except (OSError, IOError):
                # If no files left, put pointer beyond end of last file.
                if file_nr != len(self._file_sizes):  # pragma: no cover
                    raise
                self._open(file_nr - 1)
                break

        self.fh.seek(offset - self._file_offsets[self.file_nr])
        return self.tell()
    seek.__doc__ = io.BufferedIOBase.seek.__doc__

    def read(self, count=None):
        if self.closed:
            raise ValueError('read of closed file.')

        if count is None or count < 0:
            count = max(self.size - self.tell(), 0)

        data = b''
        while count > 0:
            extra = self.fh.read(count)
            if not extra:
                break
            count -= len(extra)
            if not data:  # avoid copies for first read.
                data = extra
            else:
                data += extra
            # Go to current offset, possibly opening new file.
            self.seek(0, 1)

        return data
    read.__doc__ = io.BufferedIOBase.read.__doc__


class SequentialFileWriter(SequentialFileBase):
    """Write several files as if they were one contiguous one.

    Note that the file is not seekable and readable.

    Parameters
    ----------
    files : list, tuple, or other iterable of str, filehandle
        The contains the names of the underlying files that should be combined.
        If not a list or tuple, it should allow indexing with positive indices
        (e.g., returning a name as derived from a template).  It should raise
        raise `IndexError` if the index is out of range.
    mode : str, optional
        The mode with which the files should be opened (default: 'w+b'). If
        this does not include '+' for reading, memory maps are not possibe.
    file_size : int, optional
        The maximum size a file is allowed to have.  Default: `None`, which
        means it is unlimited and only a single file will be written (making
        using this class somewhat pointless).
    opener : callable, optional
        Function to open a single file (default: `io.open`).
    """
    def __init__(self, files, mode='w+b', file_size=None, opener=None):
        self.file_size = file_size
        super(SequentialFileWriter, self).__init__(files, mode, opener)

    def write(self, data):
        if self.closed:
            raise ValueError('write to closed file.')
        offset0 = self.tell()
        if self.file_size is not None:
            remaining = self.file_size - self.fh.tell()
            while len(data) > remaining:
                self.fh.write(data[:remaining])
                data = data[remaining:]
                self._open(self.file_nr + 1)
                remaining = self.file_size

        self.fh.write(data)
        return self.tell() - offset0
    write.__doc__ = io.BufferedIOBase.write.__doc__

    def memmap(self, dtype=np.uint8, mode=None, offset=None, shape=None,
               order='C'):
        """Map part of the file in memory.  Cannnot span file boundaries."""
        if shape is None:
            raise ValueError('cannot make writable memmap without shape.')
        return super(SequentialFileWriter,
                     self).memmap(dtype, mode, offset, shape, order)


def open(files, mode='rb', file_size=None, opener=None):
    """Read or write several files as if they were one contiguous one.

    Parameters
    ----------
    files : list, tuple, or other iterable of str, filehandle
        Contains the names of the underlying files that should be combined,
        ordered in time.  If not a list or tuple, it should allow indexing with
        positive indices, and raise `IndexError` if these are out of range.
    mode : str, optional
        The mode with which the files should be opened (default: 'rb').
    file_size : int, optional
        For writing, the maximum size of a file, beyond which a new file should
        be opened.  Default: `None`, which means it is unlimited and only a
        single file will be written.
    opener : callable, optional
        Function to open a single file (default: `io.open`).

    Notes
    -----
    The returned reader/writer will have a ``memmap`` method with which part of
    the files can be mapped to memory (like with `~numpy.memmap`), as long as
    those parts do not span files (and the underlying files are regular ones).
    For writing, this requires opening in read-write mode (i.e., 'w+b').

    Methods other than ``read``, ``write``, ``seek``, ``tell``, and ``close``
    are tried on the underlying file.  This implies, e.g., ``readline`` is
    possible, though the line cannot span multiple files.

    The reader assumes the sequence of files is **contiguous in time**, ie.
    with no gaps in the data.
    """
    if 'r' in mode:
        if file_size is not None:
            raise TypeError("cannot pass in 'file_size' for reading.")
        return SequentialFileReader(files, mode, opener=opener)
    elif 'w' in mode:
        return SequentialFileWriter(files, mode, file_size=file_size,
                                    opener=opener)
    else:
        raise ValueError("invalid mode '{0}'".format(mode))
