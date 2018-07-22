import numpy as np
from astropy.time import Time
from ..vlbi_base.header import VLBIHeaderBase, HeaderParser

# only works for aligned datatypes
def make_parser_from_dtype(mydt):
    arglst = []
    for i in range(len(mydt)):
        dt_name = mydt.names[i]
        dt_size = mydt.fields[dt_name][0].itemsize
        dt_word = mydt.fields[dt_name][1]
        arglst.append( (dt_name, (dt_word, 0, 8 * dt_size) ) )

    return tuple(arglst)

# merges two numpy dtypes a, b
# does not handle fields with the same name
# applies no additional alignment
def merge_dtype(dta, dtb):
    names = dta.names + dtb.names
    names = list(names)
    dt_dict = dict(dtb.fields)
    dt_dict.update(dta.fields) # first argument has priority
    dt_ar = []
    for n in names:
        dt_ar.append( (n, dt_dict[n][0]) )
    return np.dtype(dt_ar)

class ASPFileHeader(VLBIHeaderBase):
    _dtype = np.dtype([('n_ds', '<i4'),
                        ('n_chan', '<i4'),
                        ('ch_bw', '<f8'),
                        ('rf', '<f8'),
                        ('band_dir', '<i4'),
                        ('psr_name', 'S12'),
                        ('dm', '<f8'),
                        ('fft_len', '<i4'),
                        ('overlap', '<i4'),
                        ('n_bins', '<i4'),
                        ('t_dump', '<f4'),
                        ('n_dump', '<i4'),
                        ('n_samp_dump', '<i8'),
                        ('imjd', '<i4'),
                        ('fmjd', '<f8'),
                        ('cal_scan', '<i4'),
                        ('scan', 'S256'),
                        ('observer', 'S256'),
                        ('proj_id', 'S256'),
                        ('comment', 'S1024'),
                        ('telescope', 'S2'),
                        ('front_end', 'S256'),
                        ('pol_mode', 'S12'),
                        ('ra', '<f8'),
                        ('dec', '<f8'),
                        ('epoch', '<f4'),
                        ('pad', 'V2')]) #manual padding hack

    _header_parser = HeaderParser(make_parser_from_dtype(_dtype))

    def __init__(self, words, verify=True):
        if words is None:
            words = np.zeros((), self._dtype)
        self._words = words
        if verify:
            self.verify()

    def verify(self):
        assert self._words.dtype == self._dtype
        # Add some more useful verification?

    def __getitem__(self, item):
        return self._words[item]

    @classmethod
    def fromfile(cls, fh, *args, **kwargs):
        return cls(np.fromfile(fh, dtype=cls._dtype, count=1), *args, **kwargs)

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


class ASPBlockHeader(VLBIHeaderBase):
    _dtype = np.dtype([('totalsize', '<i4'),
                       ('NPtsSend', '<i4'),
                       ('iMJD', '<f8'),
                       ('fMJD', '<f8'),
                       ('ipts1', '<i8'),
                       ('ipts2', '<i8'),
                       ('FreqChanNo', '<i4')])
                       #itemsize is a multiple of 4 (no padding)

    _header_parser = HeaderParser(make_parser_from_dtype(_dtype))

    def __init__(self, words, verify=True):
        # Important for "fromvalues", though not obvious you'll ever use it.
        if words is None:
            words = np.zeros((), self._dtype)
        self._words = words
        if verify:
            self.verify()

    def verify(self):
        assert self._words.dtype == self._dtype
        # Add some more useful verification?

    def __getitem__(self, item):
        return self._words[item]

    @classmethod
    def fromfile(cls, fh, *args, **kwargs):
        return cls(np.fromfile(fh, dtype=cls._dtype, count=1), *args, **kwargs)

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

class ASPHeader(VLBIHeaderBase):
    _dtype = merge_dtype(ASPFileHeader._dtype, ASPBlockHeader._dtype)
    _block_dtype = ASPBlockHeader._dtype
    _header_parser = HeaderParser(make_parser_from_dtype(_dtype))

    def __init__(self, words, has_file_header=False, verify=True):
        # Important for "fromvalues", though not obvious you'll ever use it.
        if words is None:
            words = np.zeros((), self._dtype)

        if not has_file_header:
            blockwords = words.copy()
            words = np.zeros((), self._dtype)
            for n in self._block_dtype.names:
                words[n] = blockwords[n]

        self._words = words
        self._has_file_header = has_file_header
        if verify:
            self.verify()

    def has_file_header(self):
        return self._has_file_header

    def verify(self):
        assert self._words.dtype == self._dtype
        # Add some more useful verification?

    def __getitem__(self, item):
        return self._words[item]

    @classmethod
    def fromfile(cls, fh, *args, **kwargs):
       
        # this is hackey and should be made general
        # e.g. for use with sequential files
        if fh.tell() == 0:
            words = np.fromfile(fh, dtype=cls._dtype, count=1)
            has_file_header=True
        else:
            words = np.fromfile(fh, dtype=cls._block_dtype, count=1)
            has_file_header=False

        return cls(words, has_file_header=has_file_header, *args, **kwargs)

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
