import io
from bisect import bisect


class MultiFileReader(object):
    """Read several files as if they were one contiguous one."""
    def __init__(self, files, mode='rb'):
        self.files = files
        self.nfiles = len(files)
        self.file_sizes = []
        self.file_offsets = [0]
        self.file_nr = None
        self.size = 0
        for i in range(self.nfiles):
            self.open(i)
            file_size = self.fh.seek(0, 2)
            self.file_sizes.append(file_size)
            self.size += file_size
            self.file_offsets.append(self.size)
        self.open(0)
        self.offset = 0

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
            self.fh = io.open(self.files[file_nr], 'rb')
            self.file_nr = file_nr

    def tell(self):
        return self.offset

    def seek(self, offset, whence=0):
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

    def close(self):
        self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def open(files, mode='rb'):
    if mode == 'rb':
        return MultiFileReader(files)
