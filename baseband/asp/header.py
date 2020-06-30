import numpy as np
from astropy.time import Time
from astropy import units as u

from ..base.header import ParsedHeaderBase


__all__ = ['ASPFileHeader', 'ASPHeader']


class DTypeHeaderBase(ParsedHeaderBase):
    """Base for header represented by C structs.

    The struct is captured using a `numpy.dtype`, which should be
    defined on the class as ``_dtype``.
    """
    def __init__(self, words, verify=True):
        if words is None:
            self._words = np.zeros((), self._dtype)
        else:
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
        """Read ASP Header from file.

        Arguments are the same as for class initialisation.  The header
        constructed will be immutable.
        """
        s = fh.read(cls._dtype.itemsize)
        if len(s) < cls._dtype.itemsize:
            raise EOFError('reached EOF while reading ASPFileHeader')
        words = np.ndarray(buffer=s, shape=(), dtype=cls._dtype)
        return cls(words, *args, **kwargs)

    @property
    def nbytes(self):
        return self._dtype.itemsize

    @property
    def time(self):
        """Start time."""
        return Time(self['imjd'], self['fmjd'], format='mjd')


class ASPFileHeader(DTypeHeaderBase):
    """ASP baseband format file header.

    Parameters
    ----------
    words : `~numpy.ndarray`, optional
        Header words, has to have dtype ``cls._dtype``.
    verify : bool, optional
        Whether to do basic verification of integrity.

    """

    _properties = ('time',)
    """Properties accessible/usable in initialisation."""

    _dtype = np.dtype([
        ('n_ds', '<i4'),
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
        ('_pad', 'S2')])  # align=True would align to 8 bytes.

    def verify(self):
        super().verify()
        # Tests taken from ASPFile.C
        assert self['band_dir'] in (1, -1)
        assert self['ch_bw'] <= 512.
        assert self['rf'] <= 12e4
        assert 1 <= self['n_ds'] <= 4
        assert 1 <= self['n_chan'] <= 32

    @property
    def sample_rate(self):
        return self['ch_bw'] * u.MHz

    # Override just to get updated docstring.
    time = property(DTypeHeaderBase.time.fget, doc="""
        Time from file header.

        Notes
        -----

        The dspsr ``asp_param.h`` description nots that these are NOT
        precise observation start times, but rather just rough estimates
        to check that the polycos are valid.
        """)


# block heaer class promoted to general "header" label
class ASPBlockHeader(DTypeHeaderBase):
    _dtype = np.dtype([
        ('totalsize', '<i4'),
        ('nptssend', '<i4'),
        ('imjd', '<f8'),
        ('fmjd', '<f8'),
        ('ipts1', '<i8'),
        ('ipts2', '<i8'),
        ('freqchanno', '<i4')])

    @property
    def payload_nbytes(self):
        return self['totalsize']

    @property
    def frame_nbytes(self):
        return self.payload_nbytes + self.nbytes

    @property
    def samples_per_frame(self):
        return self['nptssend']


class ASPHeader(ParsedHeaderBase):
    pass
