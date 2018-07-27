import numpy as np
from astropy.time import Time
from ..vlbi_base.header import VLBIHeaderBase

class ASPBlockHeader(VLBIHeaderBase):
    _dtype = np.dtype([('totalsize', '<i4'),
                       ('NPtsSend', '<i4'),
                       ('iMJD', '<u4'),
                       ('fMJD', '<f8'),
                       ('ipts1', '<i8'),
                       ('ipts2', '<i8'),
                       ('FreqChanNo', '<i4')])

    def __init__(self, words, verify=True):
        # Important for "fromvalues", though not obvious you'll ever use it.
        if words is None:
            words = np.zeros((), self._dtype)
        self._words = words
        if verify:
            self.verify()

    def verify(self):
        assert self.words.dtype == self._dtype
        # Add some more useful verification?

    def __getitem__(self, item):
        return self.words[item]

    @classmethod
    def fromfile(cls, fh, *args, **kwargs):
        s = fh.read(cls._dtype.itemsize)
        if len(s) != cls._dtype.itemsize:
            raise EOFError
        return cls(np.frombuffer(s, dtype=cls._dtype)[0], *args, **kwargs)

    @property
    def size(self):
        return self._dtype.itemsize

    @property
    def framesize(self):
        return self['totalsize']

    @property
    def payloadsize(self):
        return self.framesize - self.size

    @property
    def time(self):
        # Is scale UTC?
        return Time(self['iMJD'], self['fMJD'], format='mjd', scale='utc')