import numpy as np
from astropy.time import Time
from astropy import units as u

from ..base.header import ParsedHeaderBase
from ..base.utils import fixedvalue


__all__ = ['ASPFileHeader', 'ASPHeader']


class ASPHeaderBase(ParsedHeaderBase):
    """Base for header represented by C structs.

    The struct is captured using a `numpy.dtype`, which should be
    defined on the class as ``_dtype``.
    """

    _properties = ('sample_shape', 'time')

    def __init__(self, words, verify=True):
        if words is None:
            self.words = np.zeros((), self._dtype)
        else:
            self.words = words
            if verify:
                self.verify()

    def verify(self):
        assert self.words.dtype == self._dtype
        # Add some more useful verification?

    def __getitem__(self, item):
        try:
            return self.words[item]
        except ValueError:
            raise KeyError(f"{self.__class__.__name__} header does not "
                           f"contain {item}") from None

    @classmethod
    def fromfile(cls, fh, verify=True, **kwargs):
        """Read ASP Header from file.

        Arguments are the same as for class initialisation.  The header
        constructed will be immutable.
        """
        s = fh.read(cls._dtype.itemsize)
        if len(s) < cls._dtype.itemsize:
            raise EOFError('reached EOF while reading ASPFileHeader')
        words = np.ndarray(buffer=s, shape=(), dtype=cls._dtype)
        return cls(words, verify=verify, **kwargs)

    @property
    def nbytes(self):
        return self._dtype.itemsize

    @property
    def sample_shape(self):
        npol = 2  # fixed value for all payloads??
        return npol,

    @property
    def time(self):
        """Start time."""
        return Time(self['imjd'], self['fmjd'], format='mjd')

    def __eq__(self, other):
        return (type(other) is type(self)
                and other.words == self.words)


class ASPFileHeader(ASPHeaderBase):
    """ASP baseband format file header.

    Parameters
    ----------
    words : `~numpy.ndarray`, optional
        Header words, has to have dtype ``cls._dtype``.
    verify : bool, optional
        Whether to do basic verification of integrity.

    """

    _properties = ASPHeaderBase._properties + ('sample_rate',)
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
    time = property(ASPHeaderBase.time.fget, doc="""
        Time from file header.

        Notes
        -----

        The dspsr ``asp_param.h`` description nots that these are NOT
        precise observation start times, but rather just rough estimates
        to check that the polycos are valid.
        """)


class ASPHeader(ASPHeaderBase):

    _properties = ('payload_nbytes', 'frame_nbytes', 'bps', 'complex_data',
                   'samples_per_frame') + ASPHeaderBase._properties

    _dtype = np.dtype([
        ('totalsize', '<i4'),
        ('nptssend', '<i4'),
        ('imjd', '<f8'),
        ('fmjd', '<f8'),
        ('ipts1', '<i8'),
        ('ipts2', '<i8'),
        ('freqchanno', '<i4')])

    def __init__(self, words, file_header=None, verify=True):
        self.file_header = file_header
        super().__init__(words, verify=verify)

    def verify(self):
        super().verify()
        # Assume the file header has already been verified, if passed in.
        assert (self.file_header is None
                or isinstance(self.file_header, ASPFileHeader))

    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            try:
                return self.file_header[item]
            except (KeyError, TypeError):
                pass
            raise

    def __getattr__(self, attr):
        if (self.file_header is not None
                and attr in self.file_header._properties):
            return getattr(self.file_header, attr)

        return self.__getattribute__(attr)

    @property
    def payload_nbytes(self):
        return self['totalsize']

    @property
    def frame_nbytes(self):
        return self.payload_nbytes + self.nbytes

    @fixedvalue
    def bps(cls):
        """For ASP, encoding is always using 8 bits per sample."""
        return 8

    @fixedvalue
    def complex_data(cls):
        """ASP data are always complex."""
        return True

    @property
    def samples_per_frame(self):
        return self['nptssend']

    start_time = property(ASPHeaderBase.time.fget, doc=(
        """Start time of the observation."""))

    def get_time(self, frame_rate=None):
        """Time for the current block.

        Parameters
        ----------
        frame_rate : `~astropy.units.Quantity`, optional
            Used to calculate the offset from the start if it is non-zero.

        Returns
        -------
        `~astropy.time.Time`
        """
        start_time = self.start_time
        ipts1 = self['ipts1']
        if ipts1 == 0:
            return start_time

        if frame_rate is None:
            if self.file_header is None:
                raise ValueError('when no file header is attached, need a '
                                 'frame rate to calculate the time '
                                 'for a nonzero offset.')
            frame_rate = self.file_header.sample_rate / self.samples_per_frame

        return start_time + ipts1 / (self.samples_per_frame * frame_rate)

    time = property(get_time)
