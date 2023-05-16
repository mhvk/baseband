# Licensed under the GPLv3 - see LICENSE
"""
Definitions for GUPPI headers.

Implements a GUPPIHeader class that reads & writes FITS-like headers from file.
"""
import operator

import astropy.units as u
from astropy.io import fits
from astropy.time import Time, TimeDelta


__all__ = ['GUPPIHeader']


class GUPPIHeader(fits.Header):
    """GUPPI baseband file format header.

    Parameters
    ----------
    *args : str or iterable
        If a string, parsed as a GUPPI header from a file, otherwise
        as for the `astropy.io.fits.Header` baseclass.
    verify : bool, optional
        Whether to do minimal verification that the header is consistent with
        the GUPPI standard.  Default: `True`.
    mutable : bool, optional
        Whether to allow the header to be changed after initialisation.
        Default: `True`.
    **kwargs
        Any further header keywords to be set.

    Notes
    -----
    Like `~astropy.io.fits.Header`, the initialiser does not accept keyword
    arguments to populate an array - instead, one must pass an iterable. In
    order to ensure keywords are kept in the right order, one should pass on
    values as a tuple, not as a dict. E.g., to copy a header, one should not
    do ``GUPPIHeader({key: header[key] for key in header})``, but rather::

        GUPPIHeader(((key, header[key]) for key in header))

    or, to also keep the comments::

        GUPPIHeader(((key, (header[key], header.comments[key]))
                      for key in header))
    """

    _properties = ('payload_nbytes', 'frame_nbytes', 'bps', 'nchan', 'npol',
                   'sample_shape', 'sample_rate', 'sideband', 'overlap',
                   'samples_per_frame', 'offset', 'start_time',
                   'time')
    """Properties accessible/usable in initialisation for all headers."""

    _defaults = [('BACKEND', 'GUPPI'),
                 ('BLOCSIZE', 0),
                 ('STT_OFFS', 0),
                 ('PKTIDX', 0),
                 ('OVERLAP', 0),
                 ('SRC_NAME', 'unset'),
                 ('TELESCOP', 'unset'),
                 ('PKTFMT', '1SFA'),
                 ('PKTSIZE', 8192),
                 ('NBITS', 8),
                 ('NPOL', 1),
                 ('OBSNCHAN', 1)]

    supported_formats = {'1SFA', 'SIMPLE'}
    """GUPPI formats that are known to work.

    '1SFA' is used for all modes other than FAST4K (which is only used
    for total intensity).  'SIMPLE' is from DSPSR, and used to support
    time-first payloads.  See
    https://safe.nrao.edu/wiki/pub/Main/JoeBrandt/guppi_status_shmem.pdf

    If a format is not in this set, yet is known to work, a PR would be
    most welcome.
    """

    def __init__(self, *args, verify=True, mutable=True, **kwargs):
        # Comments handled by fits.Header__init__().
        super().__init__(*args, **kwargs)
        self.mutable = mutable
        # Empty header always OK, since things will be added to it.
        if len(self) and verify:
            self.verify()

    def verify(self):
        """Basic check of integrity."""
        # Same check as dspsr's dsp::GUPPIFile::is_valid
        assert all(key in self for key in ('BLOCSIZE',
                                           'PKTIDX'))
        # We could check here for self['PKTFMT'] in self.supported_formats
        # but that would break reading of unsupported but working formats,
        # so instead this becomes just a warning in file_info.

    def copy(self):
        """Create a mutable and independent copy of the header."""
        # This method exists because io.fits.Header.copy has a docstring that
        # refers to `Header` which breaks sphinx...
        return super().copy()

    @classmethod
    def fromfile(cls, fh, verify=True):
        """Reads in GUPPI header block from a file.

        Parameters
        ----------
        fh : filehandle
            To read data from.
        verify: bool, optional
            Whether to do basic checks on whether the header is valid. Verify
            is automatically called by `~astropy.io.fits.Header.fromstring`, so
            this flag exists only to standardize the API.
        """
        header_start = fh.tell()
        # Find the size of the header.  GUPPI header entries are 80 char long
        # with <=8 char keyword names.  "=" is always the 9th char.
        line = '========='
        while line[8] in ('=', ' '):
            line = fh.read(80).decode('ascii')
            if line[:3] == 'END':
                break
            if line == '':
                raise EOFError

        header_end = fh.tell()
        fh.seek(header_start)
        hdr = fh.read(header_end - header_start).decode('ascii')
        # Create the header using the base class.
        self = cls.fromstring(hdr)
        self.mutable = False
        if verify:
            self.verify()
        # GUPPI headers are not a proper FITS standard, and we're reading
        # from a file that the user likely cannot control, so let's not bother
        # with card verification (this avoids warnings in repr(); gh-282)
        for c in self.cards:
            c._verified = True
        return self

    def tofile(self, fh):
        """Write GUPPI file header to filehandle.

        Uses `~astropy.io.fits.Header.tostring`.
        """
        fh.write(self.tostring(padding=False).encode('ascii'))

    @classmethod
    def fromkeys(cls, *args, verify=True, mutable=True, **kwargs):
        """Initialise a header from keyword values.

        Like fromvalues, but without any interpretation of keywords.

        Note that this just passes kwargs to the class initializer as a dict
        (for compatibility with fits.Header). It is present for compatibility
        with other header classes only.
        """
        return cls(kwargs, *args, verify=verify, mutable=mutable)

    @classmethod
    def fromvalues(cls, **kwargs):
        """Initialise a header from parsed values.

        Here, the parsed values must be given as keyword arguments, i.e., for
        any ``header``, ``cls.fromvalues(**header) == header``.

        However, unlike for the ``fromkeys`` class method, data can also be set
        using arguments named after header methods, such as ``time``.

        Furthermore, some header defaults are set in ``GUPPIHeader._defaults``.
        """
        self = cls(cls._defaults, verify=False)
        self.update(**kwargs)
        return self

    def update(self, *, verify=True, **kwargs):
        """Update the header with new values.

        Here, any keywords matching properties are processed as well, in the
        order set by the class (in ``_properties``), and after all other
        keywords have been processed.

        Parameters
        ----------
        verify : bool, optional
            If `True` (default), verify integrity after updating.
        **kwargs
            Arguments used to set keywords and properties.
        """
        # Remove kwargs that set properties, in correct order.
        extras = [(key, kwargs.pop(key)) for key in self._properties
                  if key in kwargs]
        # Update the normal keywords.
        super().update(kwargs)
        # Now set the properties.
        for attr, value in extras:
            setattr(self, attr, value)
        if verify:
            self.verify()

    def __setitem__(self, key, value):
        if not self.mutable:
            raise TypeError("immutable {0} does not support assignment."
                            .format(type(self).__name__))

        super().__setitem__(key.upper(), value)

    @property
    def nbytes(self):
        """Size of the header in bytes."""
        return (len(self) + 1) * 80

    @property
    def payload_nbytes(self):
        """Size of the payload in bytes."""
        return int(self['BLOCSIZE'])

    @payload_nbytes.setter
    def payload_nbytes(self, payloadsize):
        self['BLOCSIZE'] = payloadsize

    @property
    def frame_nbytes(self):
        """Size of the frame in bytes."""
        return self.nbytes + self.payload_nbytes

    @frame_nbytes.setter
    def frame_nbytes(self, frame_nbytes):
        self.payload_nbytes = frame_nbytes - self.nbytes

    @property
    def bps(self):
        """Bits per elementary sample."""
        return int(self['NBITS'])

    @bps.setter
    def bps(self, bps):
        self['NBITS'] = bps

    @property
    def complex_data(self):
        """Whether the data are complex."""
        return int(self['OBSNCHAN']) != 1

    @property
    def npol(self):
        """Number of polarisations."""
        return int(self['NPOL']) // (2 if self.complex_data else 1)

    @npol.setter
    def npol(self, npol):
        self['NPOL'] = npol * (2 if self.complex_data else 1)

    @property
    def nchan(self):
        """Number of channels."""
        return int(self['OBSNCHAN'])

    @nchan.setter
    def nchan(self, nchan):
        self['OBSNCHAN'] = operator.index(nchan)

    @property
    def sample_shape(self):
        """Shape of a sample in the payload (npol, nchan)."""
        return self.npol, self.nchan

    @sample_shape.setter
    def sample_shape(self, sample_shape):
        # Need to set nchan first to properly set npol, since nchan is
        # connected to complex_data.
        self.nchan = sample_shape[1]
        self.npol = sample_shape[0]

    @property
    def _bpcs(self):
        """Bits per complete sample."""
        # NPOL includes factor of 2 for real/complex components.
        return int(self['OBSNCHAN']) * int(self['NPOL']) * self.bps

    @property
    def sample_rate(self):
        """Number of complete samples per second.

        Can be set with a negative quantity to set `sideband`.  Overlap samples
        are not included in the rate.
        """
        return (1. / float(self['TBIN'])) * u.Hz

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        self['TBIN'] = 1. / abs(sample_rate.to_value(u.Hz))
        bw = (sample_rate.to_value(u.MHz) * int(self['OBSNCHAN'])
              / (1 if self.complex_data else 2))
        self['OBSBW'] = bw

    @property
    def sideband(self):
        """True if upper sideband."""
        return float(self['OBSBW']) > 0

    @sideband.setter
    def sideband(self, sideband):
        self['OBSBW'] = (1 if sideband else -1) * abs(self['OBSBW'])

    @property
    def channels_first(self):
        """True if encoded payload ordering is (nchan, nsample, npol)."""
        # Called ``time-ordered`` in DSPSR.
        return self['PKTFMT'] != 'SIMPLE'

    @channels_first.setter
    def channels_first(self, channels_first):
        self['PKTFMT'] = '1SFA' if bool(channels_first) else 'SIMPLE'

    @property
    def samples_per_frame(self):
        """Number of complete samples in the frame, including overlap."""
        return self.payload_nbytes * 8 // self._bpcs

    @samples_per_frame.setter
    def samples_per_frame(self, samples_per_frame):
        old_payload_nbytes = self.payload_nbytes
        self.payload_nbytes = (samples_per_frame * self._bpcs + 7) // 8
        if self.samples_per_frame != samples_per_frame:
            exc = ValueError("header cannot store {} samples per frame. "
                             "Nearest is {}.".format(samples_per_frame,
                                                     self.samples_per_frame))
            self.payload_nbytes = old_payload_nbytes
            raise exc

    @property
    def overlap(self):
        """Number of complete samples that overlap with the next frame."""
        return int(self['OVERLAP'])

    @overlap.setter
    def overlap(self, overlap):
        self['OVERLAP'] = operator.index(overlap)

    @property
    def offset(self):
        """Offset from start of observation in units of time."""
        # PKTIDX only counts valid packets, not overlap ones.
        return ((self['PKTIDX'] * self['PKTSIZE'] * 8 // self._bpcs)
                * float(self['TBIN'])) * u.s

    @offset.setter
    def offset(self, offset):
        self['PKTIDX'] = int((offset / (float(self['TBIN']) * u.s)
                              / self['PKTSIZE']
                              * ((self._bpcs + 7) // 8)).to(u.one).round())

    @property
    def start_time(self):
        """Start time of the observation."""
        return (Time(self['STT_IMJD'], scale='utc', format='mjd')
                + TimeDelta(self['STT_SMJD'], self['STT_OFFS'], format='sec'))

    @start_time.setter
    def start_time(self, start_time):
        start_time = Time(start_time, scale='utc')
        imjd = int(start_time.mjd)
        # Calculate differences from start of day.
        djd = start_time - Time(imjd, format='mjd', scale='utc')
        # Correct for possible rounding errors.
        imjd += int(djd.jd)
        # Get seconds.  Should now be guaranteed to be between 0 and 86400
        # (or 86401 if a leap-second day).
        seconds = (start_time - Time(imjd, format='mjd', scale='utc')).sec
        self['STT_IMJD'] = imjd
        self['STT_SMJD'], self['STT_OFFS'] = divmod(seconds, 1)

    @property
    def time(self):
        """Start time of the part of the observation covered by this header."""
        return self.start_time + self.offset

    @time.setter
    def time(self, time):
        """Set header time.

        If ``start_time`` is not already set, this sets it using the time and
        ``offset``.  Otherwise, this sets ``offset`` using the time and
        ``start_time``.

        Parameters
        ----------
        time : `~astropy.time.Time`
            Time for the first sample associated with this header.
        """
        if 'STT_IMJD' not in self.keys():
            self.start_time = time - self.offset
        else:
            self.offset = time - self.start_time

    def _ipython_key_completions_(self):
        # Enables tab-completion of header keys in IPython.
        return self.keys()

    def __eq__(self, other):
        """Whether headers have the same keys with the same values."""
        return all(self.get(k, None) == other.get(k, None)
                   for k in (set(self.keys()) | set(other.keys())))

    def __repr__(self):
        name = self.__class__.__name__
        vals = super().__repr__()
        return ('<{0} {1}>'.format(name, ("\n  " + len(name) * " ").join(
            [v.rstrip() for v in vals.split('\n')])))
