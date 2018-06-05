# Licensed under the GPLv3 - see LICENSE
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
from astropy.time import Time
from astropy.extern import six


__all__ = ['DADAHeader']


class DADAHeader(OrderedDict):
    """DADA baseband file format header.

    Defines a number of routines common to all baseband format headers.

    Parameters
    ----------
    *args : str or iterable
        If a string, parsed as a DADA header from a file, otherwise
        as for the OrderedDict baseclass.
    verify : bool, optional
        Whether to do minimal verification that the header is consistent with
        the DADA standard.  Default: `True`.
    mutable : bool, optional
        Whether to allow the header to be changed after initialisation.
        Default: `True`.
    **kwargs
        Any further header keywords to be set.  If any value is a 2-item tuple,
        the second one will be considered a comment.

    Notes
    -----
    Like `~collections.OrderedDict`, in order to ensure keywords are kept in
    the right order, one should pass on values as a tuple, not as a dict.
    E.g., to copy a header, one should not do ``DADAHeader(**header)``, but
    rather::

        DADAHeader(((key, header[key]) for key in header))

    or, to also keep the comments::

        DADAHeader(((key, (header[key], header.comments[key]))
                   for key in header))
    """

    _properties = ('payload_nbytes', 'frame_nbytes', 'bps', 'complex_data',
                   'sample_shape', 'sample_rate', 'sideband',
                   'tsamp', 'samples_per_frame', 'offset', 'start_time',
                   'time')
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
        self.mutable = True
        self.comments = {}
        if len(args) == 1 and isinstance(args[0], six.string_types):
            args = (self._fromlines(args[0].split('\n')),)

        super(DADAHeader, self).__init__(*args, **kwargs)
        self.mutable = mutable
        if verify:
            self.verify()

    def verify(self):
        """Basic check of integrity."""
        assert self['HEADER'] == 'DADA'
        assert all(key in self for key in ('HDR_VERSION',
                                           'HDR_SIZE',
                                           'DADA_VERSION'))

    def copy(self):
        """Create a mutable and independent copy of the header."""
        # Cannot do super(DADAHeader, self).copy(), since this first
        # initializes an empty header, which does not pass verification.
        new = self.__class__(self)
        new.comments = self.comments.copy()
        new.mutable = True
        return new

    def __copy__(self):
        return self.copy()

    @staticmethod
    def _fromlines(lines):
        """Interpret a list of lines as a header, converting its values."""
        args = []
        for line_no, line in enumerate(lines):
            split = line.strip().split('#')
            comment = split[1].strip() if (len(split) > 1 and
                                           split[1]) else None
            split = split[0].strip().split() if split else []
            key = split[0] if split and split[0] else '_{0:d}'.format(line_no)
            value = split[1] if (len(split) > 1 and split[1]) else None

            if key in ('FILE_SIZE', 'FILE_NUMBER', 'HDR_SIZE',
                       'OBS_OFFSET', 'OBS_OVERLAP',
                       'NBIT', 'NDIM', 'NPOL', 'NCHAN', 'RESOLUTION', 'DSB'):
                value = int(value)

            elif key in ('FREQ', 'BW', 'TSAMP'):
                value = float(value)

            args.append((key, (value, comment)))

        return args

    def _tolines(self):
        """Write header to a list of strings."""
        lines = []
        for key in self:
            value = self[key]
            comment = self.comments.get(key, None)
            if value is not None:
                if comment is not None:
                    line = '{0} {1} # {2}'.format(key, value, comment)
                else:
                    line = '{0} {1}'.format(key, value)
            else:
                if comment is not None:
                    line = '# {0}'.format(comment)
                else:
                    line = ''
            lines.append(line)
        return lines

    @classmethod
    def fromfile(cls, fh, verify=True):
        """
        Reads in DADA header block from a file.

        The file pointer should be at the start.

        Parameters
        ----------
        fh : filehandle
            To read data from.
        verify: bool, optional
            Whether to do basic checks on whether the header is valid.
            Default: `True`.
        """
        start_pos = fh.tell()
        hdr_size = 4096
        lines = []
        while fh.tell() - start_pos < hdr_size:
            line = fh.readline().decode('ascii')
            if line == '' or line[0] == '#' and 'end of header' in line:
                break

            if line.startswith('HDR_SIZE'):
                hdr_size = int(line.split()[1])

            lines.append(line)

        if fh.tell() - start_pos > hdr_size:
            warnings.warn("Odd, read {0} bytes while the header size is {1}"
                          .format(fh.tell(), hdr_size))
        else:
            fh.seek(start_pos + hdr_size)

        return cls(cls._fromlines(lines), verify=verify, mutable=False)

    def tofile(self, fh):
        """Write DADA file header to filehandle.

        Parts of the header beyond the ascii lines are filled with 0x00.
        Note that file should in principle be at the start, but we don't check
        for that since that would break SequentialFileWriter.
        """
        start_pos = fh.tell()
        with io.BytesIO() as s:
            for line in self._tolines():
                s.write((line + '\n').encode('ascii'))
            s.write('# end of header\n'.encode('ascii'))
            extra = self.nbytes - s.tell()
            if extra < 0:
                raise ValueError("cannot write header in allocated size of "
                                 "{0}".format(self.nbytes))
            s.seek(0)
            fh.write(s.read())
            if extra:
                fh.write(b'\00' * extra)
            assert fh.tell() - start_pos == self.nbytes

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
        using arguments named after header methods, such as ``time``.

        Furthermore, some header defaults are set in ``DADAHeader._defaults``.
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
        # Remove kwargs that set properties, in correct order.
        extras = [(key, kwargs.pop(key)) for key in self._properties
                  if key in kwargs]
        # Update the normal keywords.
        super(DADAHeader, self).update(**kwargs)
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

        super(DADAHeader, self).__setitem__(key.upper(), value)

    @property
    def nbytes(self):
        """Size of the header in bytes."""
        return self['HDR_SIZE']

    @property
    def payload_nbytes(self):
        """Size of the payload in bytes."""
        return self['FILE_SIZE']

    @payload_nbytes.setter
    def payload_nbytes(self, payload_nbytes):
        self['FILE_SIZE'] = payload_nbytes

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
        return self['NBIT']

    @bps.setter
    def bps(self, bps):
        self['NBIT'] = bps

    @property
    def complex_data(self):
        """Whether the data are complex."""
        return self['NDIM'] == 2

    @complex_data.setter
    def complex_data(self, complex_data):
        self['NDIM'] = 2 if complex_data else 1

    @property
    def sample_shape(self):
        """Shape of a sample in the payload (npol, nchan)."""
        return self['NPOL'], self['NCHAN']

    @sample_shape.setter
    def sample_shape(self, sample_shape):
        self['NPOL'], self['NCHAN'] = sample_shape

    @property
    def sample_rate(self):
        """Number of complete samples per second.

        Can be set with a negative quantity to set `sideband`.
        """
        return (1. / self['TSAMP']) * u.MHz

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        sample_rate = sample_rate.to_value(u.MHz)
        self['TSAMP'] = 1. / abs(sample_rate)
        bw = sample_rate * self['NCHAN'] / (1 if self.complex_data else 2)
        self['BW'] = (-1 if self.get('BW', bw) < 0 else 1) * bw

    @property
    def sideband(self):
        """True if upper sideband."""
        return self['BW'] > 0

    @sideband.setter
    def sideband(self, sideband):
        self['BW'] = (1 if sideband else -1) * abs(self['BW'])

    @property
    def samples_per_frame(self):
        """Number of complete samples in the frame."""
        return (self.payload_nbytes * 8 //
                self.bps // (2 if self.complex_data else 1) // self['NPOL'] //
                self['NCHAN'])

    @samples_per_frame.setter
    def samples_per_frame(self, samples_per_frame):
        self.payload_nbytes = (
            (samples_per_frame * self['NCHAN'] * self['NPOL'] *
             self.bps * (2 if self.complex_data else 1) + 7) // 8)

    @property
    def offset(self):
        """Offset from start of observation in units of time."""
        return ((self['OBS_OFFSET'] *
                 8 // (self['NBIT'] * self['NDIM'] *
                       self['NPOL'] * self['NCHAN'])) *
                self['TSAMP'] * u.us)

    @offset.setter
    def offset(self, offset):
        self['OBS_OFFSET'] = (int(round(offset.to(u.us).value /
                                        self['TSAMP'])) *
                              ((self['NBIT'] * self['NDIM'] *
                                self['NPOL'] * self['NCHAN'] + 7) // 8))

    @property
    def start_time(self):
        """Start time of the observation."""
        mjd_int, frac = self['MJD_START'].split('.')
        mjd_int = int(mjd_int)
        frac = float('.' + frac)
        # replace '-' between date and time with a 'T' and convert to Time
        return Time(mjd_int, frac, scale='utc', format='mjd')

    @start_time.setter
    def start_time(self, start_time):
        start_time = Time(start_time, scale='utc', format='isot', precision=9)
        self['UTC_START'] = (start_time.isot.replace('T', '-')
                             .replace('.000000000', ''))
        mjd_int = int(start_time.mjd)
        mjd_frac = (start_time - Time(mjd_int, format='mjd',
                                      scale=start_time.scale)).jd
        if mjd_frac < 0:
            mjd_int -= 1
            mjd_frac += 1.
        self['MJD_START'] = ('{0:05d}'.format(mjd_int) +
                             ('{0:17.15f}'.format(mjd_frac))[1:])

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
        if 'MJD_START' not in self.keys():
            self.start_time = time - self.offset
        else:
            self.offset = time - self.start_time

    def _ipython_key_completions_(self):
        # Enables tab-completion of header keys in IPython.
        return self.keys()

    def __eq__(self, other):
        """Whether headers have the same keys with the same values."""
        # We do a float conversion for MJD_START, since headers often give
        # more digits than can really be stored.
        return (all(self.get(k, None) == other.get(k, None)
                    for k in (set(self.keys()) | set(other.keys()))
                    if not k.startswith('_') and k != 'MJD_START') and
                float(self.get('MJD_START', 0.)) ==
                float(other.get('MJD_START', 0.)))

    def __repr__(self):
        return('{0}("""'.format(self.__class__.__name__) +
               '\n'.join(self._tolines()) +
               '""")')
