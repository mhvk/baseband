# Licensed under the GPLv3 - see LICENSE.rst
"""
Definitions for DADA pulsar baseband headers.

Implements a DADAHeader class used to store header definitions in a FITS
header, and read & write these from files.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import io
import warnings
from collections import OrderedDict
import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.extern import six


__all__ = ['GUPPIHeader']


class GUPPIHeader(fits.Header):
    """GUPPI baseband file format header."""

    _properties = ('payloadsize', 'framesize', 'bps', 'complex_data',
                   'sample_shape', 'bandwidth', 'sideband', 'tsamp',
                   'samples_per_frame', 'offset', 'time0', 'time')
    """Properties accessible/usable in initialisation for all headers."""

    _defaults = [('HEADER', 'DADA'),
                 ('HDR_VERSION', '1.0'),
                 ('HDR_SIZE', 4096),
                 ('DADA_VERSION', '1.0'),
                 ('OBS_ID', 'unset'),
                 ('PRIMARY', 'unset'),
                 ('SECONDARY', 'unset'),
                 ('FILE_NAME', 'unset'),
                 ('FILE_NUMBER', 0),
                 ('OBS_OFFSET', 0),
                 ('OBS_OVERLAP', 0),
                 ('SOURCE', 'unset'),
                 ('TELESCOPE', 'unset'),
                 ('INSTRUMENT', 'unset'),
                 ('RECEIVER', 'unset'),
                 ('NBIT', 8),
                 ('NDIM', 1),
                 ('NPOL', 1),
                 ('NCHAN', 1),
                 ('RESOLUTION', 1),
                 ('DSB', 1)]

    def __init__(self, *args, **kwargs):
        verify = kwargs.pop('verify', True)
        mutable = kwargs.pop('mutable', True)
        super(GUPPIHeader, self).__init__(*args, **kwargs)
        self.mutable = mutable
        if verify:
            self.verify()

    def verify(self):
        """Basic check of integrity."""
        assert all(key in self for key in ('BLOCSIZE',
                                           'PKTIDX'))
        assert self['PKTFMT'] == '1SFA'

    @classmethod
    def fromfile(cls, fh, verify=True):
        """
        Reads in GUPPI header block from a file.

        The file pointer should be at the start.

        Parameters
        ----------
        fh : filehandle
            To read data from.
        verify: bool
            Whether to do basic checks on whether the header is valid.
        """
        header_start = fh.tell()
        line = '========='
        while line[8] == '=':
            line = fh.read(80).decode('ascii')
            if line[:3] == 'END':
                break
        
        header_end = fh.tell()
        fh.seek(header_start)
        hdr = fh.read(header_end-header_start).decode('ascii')
        fits_header = fits.Header.fromstring(hdr)
        return cls(fits_header, verify=verify, mutable=False)

    def tofile(self, fh):
        """Write GUPPI file header to filehandle.

        Parts of the header beyond the ascii lines are filled with 0x00.
        Note that file should be at the start.
        """
        raise NotImplementedError

    @classmethod
    def fromkeys(cls, *args, **kwargs):
        """Initialise a header from keyword values.

        Like fromvalues, but without any interpretation of keywords.

        This just calls the class initializer; it is present for compatibility
        with other header classes only.
        """
        return cls(*args, **kwargs)

    @classmethod
    def fromvalues(cls, **kwargs):
        """Initialise a header from parsed values.

        Here, the parsed values must be given as keyword arguments, i.e., for
        any ``header``, ``cls.fromvalues(**header) == header``.

        However, unlike for the ``fromkeys`` class method, data can also be set
        using arguments named after header methods such as ``time``.

        Furthermore, some header defaults are set in ``GUPPIHeader._defaults``.
        """
        self = cls(cls._defaults, verify=False)
        self.update(**kwargs)
        return self

    def update(self, **kwargs):
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
        verify = kwargs.pop('verify', True)
        # remove kwargs that set properties, in correct order.
        extras = [(key, kwargs.pop(key)) for key in self._properties
                  if key in kwargs]
        # update the normal keywords.
        super(GUPPIHeader, self).update(**kwargs)
        # Now set the properties.
        for attr, value in extras:
            setattr(self, attr, value)
        if verify:
            self.verify()

    def __setitem__(self, key, value):
        if not self.mutable:
            raise TypeError("immutable {0} does not support assignment."
                            .format(type(self).__name__))
        if isinstance(value, tuple):
            value, comment = value
            self.comments[key.upper()] = comment

        super(GUPPIHeader, self).__setitem__(key.upper(), value)

    def __getattr__(self, attr):
        """Get attribute, or, failing that, try to get key from header."""
        try:
            # Note that OrderDict does not have __getattr__
            return super(GUPPIHeader, self).__getattribute__(attr)
        except AttributeError as exc:
            try:
                return self[attr.upper()]
            except:
                raise exc

    @property
    def size(self):
        """Size in bytes of the header."""
        return (len(self) + 1) * 80

    @property
    def payloadsize(self):
        """Size in bytes of the payload part of the file."""
        return self['BLOCSIZE']

    @payloadsize.setter
    def payloadsize(self, payloadsize):
        self['BLOCSIZE'] = payloadsize

    @property
    def framesize(self):
        """Size in bytes of the full file, header plus payload."""
        return self.payloadsize + self.size

    @framesize.setter
    def framesize(self, framesize):
        self.payloadsize = framesize - self.size

    @property
    def bps(self):
        """Bits per sample (or real/imaginary part)."""
        return self['NBITS']

    @bps.setter
    def bps(self, bps):
        self['NBITS'] = bps

    @property
    def complex_data(self):
        return self['OBSNCHAN'] != 1

    @complex_data.setter
    def complex_data(self, complex_data):
        raise NotImplementedError
    # self['NDIM'] = 2 if complex_data else 1

    @property
    def npol(self):
        return self['NPOL'] // (2 if self.complex_data else 1)

    @npol.setter
    def npol(self, npol):
        self['NPOL'] = npol * (2 if self.complex_data else 1)

    @property
    def sample_shape(self):
        """Shape of a single payload sample: (nchan, npol)."""
        return self['OBSNCHAN'], self.npol

    @sample_shape.setter
    def sample_shape(self, sample_shape):
        self['OBSNCHAN'], self.npol = sample_shape

    @property
    def bandwidth(self):
        """Bandwidth covered by the data."""
        return abs(self['OBSBW']) * u.MHz

    @bandwidth.setter
    def bandwidth(self, bw):
        bw = bw.to(u.MHz).value
        self['OBSBW'] = (-1 if self.get('BW', bw) < 0 else 1) * bw
        self['TBIN'] = self['OBSNCHAN'] / bw

    @property
    def sideband(self):
        """True if upper sideband."""
        return self['BW'] > 0

    @sideband.setter
    def sideband(self, sideband):
        self['BW'] = (1 if sideband else -1) * abs(self['BW'])

    @property
    def time_ordered(self):
        return self['PKTFMT'] != 'SIMPLE'

    @property
    def samples_per_frame(self):
        """Complete samples per frame (i.e., each having ``sample_shape``)."""
        return (self.payloadsize * 8 //
                self.bps // self['NPOL'] // self['OBSNCHAN'])

    @samples_per_frame.setter
    def samples_per_frame(self, samples_per_frame):
        self.payloadsize = (
            (samples_per_frame * self['OBSNCHAN'] * self['NPOL'] *
             self.bps + 7) // 8)

    @property
    def offset(self):
        """Offset from start of observation in units of time."""
        return (self['STT_OFFS'] + (self['PKTIDX'] * self['PKTSIZE'] * 8 //
                self.bps // self['NPOL'] //
                self['OBSNCHAN']) * self['TBIN'] * u.s)

    @offset.setter
    def offset(self, offset):
        raise NotImplementedError

    @property
    def time0(self):
        """Start time of the observation."""
        return (Time(self['STT_IMJD'], scale='utc', format='mjd') +
                self['STT_SMJD'] * u.s)

    @time0.setter
    def time0(self, time0):
        raise NotImplementedError

    @property
    def time(self):
        """Start time the part of the observation covered by this header."""
        return self.time0 + self.offset

    @time.setter
    def time(self, time):
        """Set header time.

        Note that this sets the start time of the header, assuming the offset
        is already set correctly.

        Parameters
        ----------
        time : `~astropy.time.Time`
            Time for the first sample associated with this header.
        """
        self.time0 = time - self.offset

    def __eq__(self, other):
        """Whether headers have the same keys with the same values."""
        # We do a float conversion for MJD_START, since headers often give
        # more digits than can really be stored.
        return all(self.get(k, None) == other.get(k, None)
                   for k in (set(self.keys()) | set(other.keys())))
