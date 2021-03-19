import numpy as np
from astropy.time import Time
from astropy import units as u

from ..base.header import ParsedHeaderBase
from ..base.utils import fixedvalue


__all__ = ['KotekanHeader']


class KotekanHeader(ParsedHeaderBase):
    """Base for header represented by C structs.

    The struct is captured using a `numpy.dtype`, which should be
    defined on the class as ``_dtype``.
    """

    _dtype = np.dtype(
        [("fpga_seq_start", "<u8"),  # sample number at start of frame.
         ("ctime", [("tv", "<i8"), ("tv_nsec", "<u8")]),  # unix time
         ("stream_id", "<u8"),  # ???
         ("dataset_id", "(2,)<u8"),  # same for all
         ("beam_number", "<u4"),  # beam 11
         ("ra", "<f4"),  # 2 different RA?
         ("dec", "<f4"),  # 2 different dec?
         ("scaling", "<u4"),  # all 48
         ("frequency_bin", "<u4"),  # between 0 and 1023.
         ], align=True)
    _properties = ('sample_shape', 'time')
    _properties = ('payload_nbytes', 'frame_nbytes', 'bps', 'complex_data',
                   'samples_per_frame', 'sample_shape', 'sample_rate', 'time')

    def __init__(self, words, verify=True):
        if words is None:
            self.words = np.zeros((), self._dtype)
        else:
            self.words = words
            if verify:
                self.verify()

    def verify(self):
        assert self.words.dtype == self._dtype

    def keys(self):
        return self._dtype.names

    @property
    def mutable(self):
        """Whether the header can be modified."""
        return self.words.flags['WRITEABLE']

    @mutable.setter
    def mutable(self, mutable):
        self.words.flags['WRITEABLE'] = mutable

    def __getitem__(self, item):
        try:
            return self.words[item]
        except ValueError:
            raise KeyError(f"{self.__class__.__name__} header does not "
                           f"contain {item}") from None

    def __setitem__(self, item, value):
        try:
            self.words[item] = value
        except ValueError:
            if not self.mutable:
                raise TypeError("header is immutable. Set '.mutable` attribute"
                                " or make a copy.")
            elif item not in self.words.dtype.names:
                raise KeyError(f"{self.__class__.__name__} header does not "
                               f"contain {item}") from None

            else:
                raise

    @classmethod
    def fromfile(cls, fh, verify=True, **kwargs):
        """Read from file.

        Arguments are the same as for class initialisation.  The header
        constructed will be immutable.
        """
        s = fh.read(cls._dtype.itemsize)
        if len(s) < cls._dtype.itemsize:
            raise EOFError('reached EOF while reading ASPFileHeader')
        words = np.ndarray(buffer=s, shape=(), dtype=cls._dtype)
        self = cls(words, verify=verify, **kwargs)
        self.mutable = False
        return self

    def tofile(self, fh):
        """Write header to filehandle."""
        return fh.write(self.words.tobytes())

    @property
    def nbytes(self):
        return self._dtype.itemsize

    @fixedvalue
    def sample_shape(cls):
        npol = 2
        return npol,

    @property
    def time(self):
        """Start time."""
        return Time(self['ctime']['tv'], self['ctime']['tv_nsec']*1e-9,
                    format='unix')

    def __eq__(self, other):
        return (type(other) is type(self)
                and other.words == self.words)

    def invariant_pattern(self):
        mask = self.__class__(None)
        for key in {'dateset_id', 'beam_number', 'scaling'}:
            mask[key] = -1
        return (np.atleast_1d(self.words).view('<u4'),
                np.atleast_1d(mask.words).view('<u4'))

    @fixedvalue
    def payload_nbytes(self):
        return 49152 * 2

    @property
    def frame_nbytes(self):
        return self.payload_nbytes + self.nbytes

    @fixedvalue
    def bps(cls):
        return 4

    @fixedvalue
    def complex_data(cls):
        return True

    @fixedvalue
    def samples_per_frame(self):
        return 49152

    @fixedvalue
    def sample_rate(self):
        return 8e8 / 2048 / u.s
