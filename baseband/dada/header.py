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
from astropy.time import Time
from astropy.extern import six


class DADAHeader(OrderedDict):
    """DADA baseband file format header.

    Defines a number of routines common to all baseband format headers.

    Parameters
    ----------
    *args : str or iterable
        If a string, parsed as a DADA header from a file, otherwise
        as for the OrderedDict baseclass.
    verify : bool
        Whether to do minimal verification that the header is consistent with
        the DADA standard
    mutable : bool
        Whether to allow the header to be changed after initialisation.
    **kwargs
        Any further header keywords to be set.  If any value is a 2-item tuple,
        the second one will be considered a comment.

    Remarks
    -------
    Like OrderedDict, in order to ensure keywords are kept in the right order,
    one should pass on values as a tuple, not as a dict.  E.g., to copy a
    header, one should not do ``DADAHeader(**header)``, but rather::

    DADAHeader(((key, header[key]) for key in header))

    or, to also keep the comments::

    DADAHeader(((key, (header[key], header.comments[key])) for key in header))
    """

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

        Parameters
        ----------
        fh : filehandle
            To read data from.
        verify: bool
            Whether to do basic checks on whether the header is valid.
        """
        if fh.tell() > 0:
            raise ValueError("DADA header should be at the start of a file.")

        hdr_size = 4096
        lines = []
        while fh.tell() < hdr_size:
            line = fh.readline().decode('ascii')
            if line == '' or line[0] == '#' and 'end of header' in line:
                break

            if line.startswith('HDR_SIZE'):
                hdr_size = int(line.split()[1])

            lines.append(line)

        if fh.tell() > hdr_size:
            warnings.warn("Odd, read {0} bytes while the header size is {1}"
                          .format(fh.tell(), hdr_size))
        else:
            fh.seek(hdr_size)

        return cls(cls._fromlines(lines), verify=verify, mutable=False)

    def tofile(self, fh):
        """Write DADA file header to filehandle.

        Parts of the header beyond the ascii lines are filled with 0x00."""
        if fh.tell() > 0:
            raise ValueError("should write header at start of file.")
        with io.BytesIO() as s:
            for line in self._tolines():
                s.write((line + '\n').encode('ascii'))
            s.write('# end of header\n'.encode('ascii'))
            extra = self.size - s.tell()
            if extra < 0:
                raise ValueError("cannot write header in allocated size of "
                                 "{0}".format(self.size))
            s.seek(0)
            fh.write(s.read())
            if extra:
                fh.write(b'\00' * extra)
            assert fh.tell() == self.size

    @classmethod
    def fromkeys(cls, *args, **kwargs):
        """Initialise a header from keyword values.

        Like fromvalues, but without any interpretation of keywords.

        For compatibility with other header classes; just calls ``__init__``.
        """
        return cls(*args, **kwargs)

    @classmethod
    def fromvalues(cls, **kwargs):
        """Initialise a header from parsed values.

        Here, the parsed values must be given as keyword arguments, i.e., for
        any ``header``, ``cls.fromvalues(**header) == header``.

        However, unlike for the ``fromkeys`` class method, data can also be set
        using arguments named after header methods such as ``time``.

        Furthermore, some header defaults are set in ``DADAHeader._defaults``.
        """
        self = cls(cls._defaults, verify=False)
        self.update(**kwargs)
        return self

    def update(self, **kwargs):
        """Update the header with new values.

        Here, any keywords matching properties are processed as well, in the
        order set by the class (in ``_properties``).

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

    def __getattr__(self, attr):
        """Get attribute, or, failing that, try to get key from header."""
        try:
            # Note that OrderDict does not have __getattr__
            return super(DADAHeader, self).__getattribute__(attr)
        except AttributeError as exc:
            try:
                return self[attr.upper()]
            except:
                raise exc

    @property
    def size(self):
        return self['HDR_SIZE']

    @property
    def payloadsize(self):
        return self['FILE_SIZE']

    @payloadsize.setter
    def payloadsize(self, payloadsize):
        self['FILE_SIZE'] = payloadsize

    @property
    def framesize(self):
        return self.size + self.payloadsize

    @framesize.setter
    def framesize(self, framesize):
        self.payloadsize = framesize - self.size

    @property
    def bps(self):
        """Bits per sample (or real/imaginary part)."""
        return self['NBIT']

    @bps.setter
    def bps(self, bps):
        self['NBIT'] = bps

    @property
    def complex_data(self):
        return self['NDIM'] == 2

    @complex_data.setter
    def complex_data(self, complex_data):
        self['NDIM'] = 2 if complex_data else 1

    @property
    def sample_shape(self):
        """Shape of a single payload sample: (npol, nchan)."""
        return self['NPOL'], self['NCHAN']

    @sample_shape.setter
    def sample_shape(self, sample_shape):
        self['NPOL'], self['NCHAN'] = sample_shape

    @property
    def bandwidth(self):
        return abs(self['BW']) * u.MHz

    @bandwidth.setter
    def bandwidth(self, bw):
        bw = bw.to(u.MHz).value
        self['BW'] = (-1 if self.get('BW', bw) < 0 else 1) * bw
        self['TSAMP'] = self['NCHAN'] / (1 if self.complex_data else 2) / bw

    @property
    def sideband(self):
        """True if upper sideband."""
        return self['BW'] > 0

    @sideband.setter
    def sideband(self, sideband):
        self['BW'] = (1 if sideband else -1) * abs(self['BW'])

    @property
    def samples_per_frame(self):
        """Complete samples per frame (i.e., each having ``sample_shape``)."""
        return (self.payloadsize * 8 //
                self.bps // (2 if self.complex_data else 1) // self['NPOL'] //
                self['NCHAN'])

    @samples_per_frame.setter
    def samples_per_frame(self, samples_per_frame):
        self.payloadsize = (
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
    def time0(self):
        mjd_int, frac = self['MJD_START'].split('.')
        mjd_int = int(mjd_int)
        frac = float('.' + frac)
        # replace '-' between date and time with a 'T' and convert to Time
        return Time(mjd_int, frac, scale='utc', format='mjd')

    @time0.setter
    def time0(self, time0):
        time0 = Time(time0, scale='utc', format='isot', precision=9)
        self['UTC_START'] = (time0.isot.replace('T', '-')
                             .replace('.000000000', ''))
        mjd_int = int(time0.mjd)
        mjd_frac = (time0 - Time(mjd_int, format='mjd', scale=time0.scale)).jd
        if mjd_frac < 0:
            mjd_int -= 1
            mjd_frac += 1.
        self['MJD_START'] = ('{0:05d}'.format(mjd_int) +
                             ('{0:17.15f}'.format(mjd_frac))[1:])

    @property
    def time(self):
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
        return (all(self.get(k, None) == other.get(k, None)
                    for k in (set(self.keys()) | set(other.keys()))
                    if not k.startswith('_') and k != 'MJD_START') and
                float(self.get('MJD_START', 0.)) ==
                float(other.get('MJD_START', 0.)))

    def __repr__(self):
        return('{0}("""'.format(self.__class__.__name__) +
               '\n'.join(self._tolines()) +
               '""")')
