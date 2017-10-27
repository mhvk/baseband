"""
Definitions for VLBI VDIF Headers.

Implements a VDIFHeader class used to store header words, and decode/encode
the information therein.

For the VDIF specification, see http://www.vlbi.org/vdif
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import astropy.units as u
from astropy.extern import six

from astropy.time import Time, TimeDelta

from ..vlbi_base.header import (four_word_struct, eight_word_struct,
                                HeaderParser, VLBIHeaderBase)
from ..mark5b.header import Mark5BHeader


__all__ = ['VDIFHeader', 'VDIFBaseHeader', 'VDIFSampleRateHeader',
           'VDIFLegacyHeader', 'VDIFHeader0', 'VDIFHeader1',
           'VDIFHeader2', 'VDIFHeader3', 'VDIFMark5BHeader',
           'VDIF_HEADER_CLASSES']


ref_max = int(2. * (Time.now().jyear - 2000.)) + 1
ref_epochs = Time(['{y:04d}-{m:02d}-01'.format(y=2000 + ref // 2,
                                               m=1 if ref % 2 == 0 else 7)
                   for ref in range(ref_max)], format='isot', scale='utc',
                  precision=9)


VDIF_HEADER_CLASSES = {}
"""Dict for storing VDIF header class definitions, indexed by their EDV."""


class VDIFHeaderMeta(type):
    """
    Registry of VDIF Header EDV types, using the ``VDIF_HEADER_CLASSES``
    dict.  Checks for keyword and subclass conflicts before registering.

    This metaclass automatically registers any subclass of
    ``~baseband.vdif.header.VDIFHeader``.  This feature can be used to insert
    new EDV types into ``VDIF_HEADER_CLASSES``; see the :ref:`New VDIF EDV
    <new_edv>` tutorial.
    """
    _registry = VDIF_HEADER_CLASSES

    def __init__(cls, name, bases, dct):

        # Ignore VDIFHeader and VDIFBaseHeader, register others
        if name not in ('VDIFHeader', 'VDIFBaseHeader',
                        'VDIFSampleRateHeader'):

            # Extract edv from class; convert to -1 if edv == False
            # for VDIFLegacy
            edv = cls._edv
            if edv is False:
                edv = -1

            # Check if EDV is already registered (edv == None is invalid).
            if edv is None:
                raise ValueError("EDV cannot be None.  It should be "
                                 "overridden by the subclass.")
            elif edv in VDIFHeaderMeta._registry:
                raise ValueError("EDV {0} already registered in "
                                 "VDIF_HEADER_CLASSES".format(edv))

            VDIFHeaderMeta._registry.update({edv: cls})

        super(VDIFHeaderMeta, cls).__init__(name, bases, dct)


@six.add_metaclass(VDIFHeaderMeta)
class VDIFHeader(VLBIHeaderBase):
    """VDIF Header, supporting different Extended Data Versions.

    Will initialize a header instance appropriate for a given EDV.
    See http://www.vlbi.org/vdif/docs/VDIF_specification_Release_1.1.1.pdf

    Parameters
    ----------
    words : tuple of int, or None
        Eight (or four for legacy VDIF) 32-bit unsigned int header words.
        If ``None``, set to a tuple of zeros for later initialisation.
    edv : int, False, or None
        Extended data version.  If `False`, a legacy header is used.
        If `None` (default), it is determined from the header.  (Given it
        explicitly is mostly useful for a slight speed-up.)
    verify : bool
        Whether to do basic verification of integrity.  Default: `True`.

    Returns
    -------
    header : `VDIFHeader` subclass
        As appropriate for the extended data version.
    """

    _properties = ('framesize', 'payloadsize', 'bps', 'nchan',
                   'samples_per_frame', 'station', 'time')
    """Properties accessible/usable in initialisation for all VDIF headers."""

    _edv = None
    _struct = eight_word_struct

    def __new__(cls, words, edv=None, verify=True, **kwargs):
        # We use edv to define which class we return.
        if edv is None:
            # If not given, we extract edv from the header words.  This uses
            # parsers defined below, in VDIFBaseHeader.
            base_parsers = VDIFBaseHeader._header_parser.parsers
            if base_parsers['legacy_mode'](words):
                edv = False
            else:
                edv = base_parsers['edv'](words)

        # Have to use key "-1" instead of "False" since the dict-lookup treats
        # 0 and False as identical.
        cls = VDIF_HEADER_CLASSES.get(edv if edv is not False else -1,
                                      VDIFBaseHeader)
        return super(VDIFHeader, cls).__new__(cls)

    def __init__(self, words, edv=None, verify=True, **kwargs):
        if edv is not None:
            self._edv = edv
        super(VDIFHeader, self).__init__(words, verify=verify, **kwargs)

    def copy(self):
        return super(VDIFHeader, self).copy(edv=self.edv)

    def same_stream(self, other):
        """Whether header is consistent with being from the same stream."""
        # EDV and most parts of words 2 and 3 should be invariant.
        return (self.edv == other.edv and
                all(self[key] == other[key]
                    for key in ('ref_epoch', 'vdif_version', 'frame_length',
                                'complex_data', 'bits_per_sample',
                                'station_id')))

    @classmethod
    def fromfile(cls, fh, edv=None, verify=True):
        """Read VDIF Header from file.

        Parameters
        ----------
        fh : filehandle
            To read data from.
        edv : int, `False`, or `None`
            Extended data version.  If `False`, a legacy header is used.
            If `None` (default), it is determined from the header.  (Given it
            explicitly is mostly useful for a slight speed-up.)
        verify : bool
            Whether to do basic verification of integrity.  Default: `True`.
        """
        # Assume non-legacy header to ensure those are done fastest.
        # Since a payload will follow, it is OK to read too many bytes even
        # for a legacy header; we just rewind below.
        s = fh.read(32)
        if len(s) != 32:
            raise EOFError
        self = cls(eight_word_struct.unpack(s), edv, verify=False)
        if self.edv is False:
            # Legacy headers are 4 words, so rewind, and remove excess data.
            fh.seek(-16, 1)
            self.words = self.words[:4]
        if verify:
            self.verify()

        return self

    @classmethod
    def fromvalues(cls, edv=False, **kwargs):
        """Initialise a header from parsed values.

        Here, the parsed values must be given as keyword arguments, i.e., for
        any ``header = cls(<data>)``, ``cls.fromvalues(**header) == header``.

        However, unlike for the :meth:`VDIFHeader.fromkeys` class method, data
        can also be set using arguments named after methods such as ``bps`` and
        ``time``.

        Given defaults for standard header keywords:

        invalid_data : `False`
        legacy_mode : `False`
        vdif_version : 1
        thread_id : 0
        frame_nr : 0

        Values set by other keyword arguments (if present):

        bits_per_sample : from ``bps``
        frame_length : from ``samples_per_frame`` or ``framesize``
        lg2_nchan : from ``nchan``
        ref_epoch, seconds, frame_nr : from ``time`` (may need ``bandwidth``)

        Given defaults for edv 1 and 3:

        sync_pattern : 0xACABFEED

        Defaults inferred from other keyword arguments for all edv:

        station_id : from ``station``
        sample_rate, sample_unit : from ``bandwidth`` or ``framerate``
        """
        # Some defaults that are not done by setting properties.
        kwargs.setdefault('legacy_mode', True if edv is False else False)
        kwargs['edv'] = edv
        return super(VDIFHeader, cls).fromvalues(edv, **kwargs)

    @classmethod
    def fromkeys(cls, **kwargs):
        """Initialise a header from parsed values.

        Like :meth:`VDIFHeader.fromvalues`, but without any interpretation of
        keywords.

        Raises
        ------
        KeyError : if not all keys required are pass in.
        """
        # Get all required values.
        edv = False if kwargs['legacy_mode'] else kwargs['edv']
        return super(VDIFHeader, cls).fromkeys(edv, **kwargs)

    @classmethod
    def from_mark5b_header(cls, mark5b_header, bps, nchan, **kwargs):
        """Construct an Mark5B over VDIF header (EDV=0xab).

        See http://www.vlbi.org/vdif/docs/vdif_extension_0xab.pdf

        Note that the Mark 5B header does not encode the bits-per-sample and
        the number of channels used in the payload, so these need to be given
        separately.  A complete frame can be encapsulated with
        VDIFFrame.from_mark5b_frame.

        Parameters
        ----------
        mark5b_header : Mark5BHeader
            Used to set time, etc.
        bps : int
            bits per sample.
        nchan : int
            Number of channels carried in the Mark 5B paylod.
        **kwargs
            Any further arguments.  Strictly, none are necessary to create a
            valid VDIF header, but this can be used to pass on, e.g.,
            ``invalid_data``.
        """
        kwargs.update(mark5b_header)
        return cls.fromvalues(edv=0xab, time=mark5b_header.time,
                              bps=bps, nchan=nchan, complex_data=False,
                              **kwargs)

    # properties common to all VDIF headers.
    @property
    def edv(self):
        """VDIF Extended Data Version (EDV)."""
        return self._edv

    @property
    def framesize(self):
        """Size of a frame, in bytes."""
        return self['frame_length'] * 8

    @framesize.setter
    def framesize(self, size):
        assert size % 8 == 0
        self['frame_length'] = int(size) // 8

    @property
    def payloadsize(self):
        """Size of the payload, in bytes."""
        return self.framesize - self.size

    @payloadsize.setter
    def payloadsize(self, size):
        self.framesize = size + self.size

    @property
    def bps(self):
        """Bits per sample (adding bits for imaginary and real for complex)."""
        return self['bits_per_sample'] + 1

    @bps.setter
    def bps(self, bps):
        assert bps % 1 == 0
        self['bits_per_sample'] = int(bps) - 1

    @property
    def nchan(self):
        """Number of channels in frame."""
        return 2**self['lg2_nchan']

    @nchan.setter
    def nchan(self, nchan):
        lg2_nchan = np.log2(nchan)
        assert lg2_nchan % 1 == 0
        self['lg2_nchan'] = int(lg2_nchan)

    @property
    def samples_per_frame(self):
        """Number of samples encoded in frame."""
        # Values are not split over word boundaries.
        values_per_word = 32 // self.bps // (2 if self['complex_data'] else 1)
        # samples are not split over payload boundaries.
        return self.payloadsize // 4 * values_per_word // self.nchan

    @samples_per_frame.setter
    def samples_per_frame(self, samples_per_frame):
        values_per_word = 32 // self.bps // (2 if self['complex_data'] else 1)
        # units of frame length are 8 bytes, i.e., 2 words.
        values_per_long = values_per_word * 2
        longs = (samples_per_frame * self.nchan - 1) // values_per_long + 1
        self['frame_length'] = int(longs) + self.size // 8

    @property
    def station(self):
        """Station ID: two ASCII characters, or 16-bit int."""
        msb = self['station_id'] >> 8
        if 48 <= msb < 128:
            return chr(msb) + chr(self['station_id'] & 0xff)
        else:
            return self['station_id']

    @station.setter
    def station(self, station):
        try:
            station_id = (ord(station[0]) << 8) + ord(station[1])
        except TypeError:
            station_id = station
        assert int(station_id) == station_id
        self['station_id'] = station_id

    def get_time(self, framerate=None, frame_nr=None):
        """
        Convert ref_epoch, seconds, and frame_nr to Time object.

        Uses 'ref_epoch', which stores the number of half-years from 2000,
        and 'seconds'.  By default, it also calculates the offset using
        the current frame number.  For non-zero frame_nr, this requires the
        frame rate, which is calculated from the header.  It can be passed on
        if this is not available (e.g., for a legacy VDIF header).

        Set frame_nr=0 to just get the header time from ref_epoch and seconds.

        Parameters
        ----------
        framerate : `~astropy.units.Quantity`, optional
            For non-zero `frame_nr`, this is used to calculate the
            corresponding offset.  If not given, it will be attempted to
            calculate it from the sampling rate given in the header (but not
            all EDV contain this).
        frame_nr : int, optional
            Can be used to override the ``frame_nr`` from the header.  If 0,
            the routine simply returns the time in the header

        Returns
        -------
        `~astropy.time.Time`
        """
        if frame_nr is None:
            frame_nr = self['frame_nr']

        if frame_nr == 0:
            offset = 0.
        else:
            if framerate is None:
                try:
                    framerate = self.framerate
                except AttributeError:
                    raise ValueError("Cannot calculate frame rate for this "
                                     "header. Pass it in explicitly.")
            offset = (frame_nr / framerate).to(u.s).value
        return (ref_epochs[self['ref_epoch']] +
                TimeDelta(self['seconds'], offset, format='sec', scale='tai'))

    def set_time(self, time, framerate=None, frame_nr=None):
        """
        Convert Time object to ref_epoch, seconds, and frame_nr.

        For non-integer seconds, the frame_nr will be calculated if not given
        explicitly. This requires the frame rate, which is calculated from the
        header.  It can be passed on if this is not available (e.g., for a
        legacy VDIF header).

        Parameters
        ----------
        time : Time instance
            The time to use for this header.
        framerate : `~astropy.units.Quantity` with frequency units, optional
            For calculating the ``frame_nr`` from the fractional seconds.
            If not given, will try to calculate it from the sampling rate
            given in the header (but not all EDV contain this).
        frame_nr : int, optional
            An explicit frame number associated with the fractions of seconds.
        """
        assert time > ref_epochs[0]
        ref_index = np.searchsorted((ref_epochs - time).sec, 0) - 1
        self['ref_epoch'] = ref_index
        seconds = time - ref_epochs[ref_index]
        int_sec = int(seconds.sec)
        if frame_nr is None:
            frac_sec = seconds - int_sec * u.s
            if abs(frac_sec) < 2. * u.ns:
                frame_nr = 0
            elif abs(1. * u.s - frac_sec) < 2. * u.ns:
                int_sec += 1
                frame_nr = 0
            else:
                if framerate is None:
                    try:
                        framerate = self.framerate
                    except AttributeError:
                        raise ValueError("Cannot calculate frame rate for "
                                         "this header. Pass it in explicitly.")
                frame_nr = int(round((frac_sec * framerate)
                                     .to(u.dimensionless_unscaled).value))
                if abs(frame_nr / framerate - 1. * u.s) < 1. * u.ns:
                    frame_nr = 0
                    int_sec += 1

        self['seconds'] = int_sec
        self['frame_nr'] = frame_nr

    time = property(get_time, set_time)


class VDIFLegacyHeader(VDIFHeader):
    """Legacy VDIF header that uses only 4 32-bit words.

    See Section 6 of
    http://www.vlbi.org/vdif/docs/VDIF_specification_Release_1.1.1.pdf
    """
    _struct = four_word_struct

    _header_parser = HeaderParser(
        (('invalid_data', (0, 31, 1, False)),
         ('legacy_mode', (0, 30, 1, True)),
         ('seconds', (0, 0, 30)),
         ('_1_30_2', (1, 30, 2, 0x0)),
         ('ref_epoch', (1, 24, 6)),
         ('frame_nr', (1, 0, 24, 0x0)),
         ('vdif_version', (2, 29, 3, 0x1)),
         ('lg2_nchan', (2, 24, 5)),
         ('frame_length', (2, 0, 24)),
         ('complex_data', (3, 31, 1)),
         ('bits_per_sample', (3, 26, 5)),
         ('thread_id', (3, 16, 10, 0x0)),
         ('station_id', (3, 0, 16))))
    # Set default
    _edv = False

    def verify(self):
        """Basic checks of header integrity."""
        assert self.edv is False
        assert self['legacy_mode']
        assert len(self.words) == 4


class VDIFBaseHeader(VDIFHeader):
    """Base for non-legacy VDIF headers that use 8 32-bit words."""

    _header_parser = VDIFLegacyHeader._header_parser + HeaderParser(
        (('legacy_mode', (0, 30, 1, False)),  # Repeat, to change default.
         ('edv', (4, 24, 8))))

    def verify(self):
        """Basic checks of header integrity."""
        assert not self['legacy_mode']
        assert self.edv is None or self.edv == self['edv']
        assert len(self.words) == 8
        if 'sync_pattern' in self.keys():
            assert (self['sync_pattern'] ==
                    self._header_parser.defaults['sync_pattern'])


class VDIFHeader0(VDIFBaseHeader):
    """VDIF Header for EDV=0.

    EDV=0 implies the extended user data fields are not used.
    """
    _edv = 0

    def verify(self):
        assert all(word == 0 for word in self.words[4:])
        super(VDIFHeader0, self).verify()


class VDIFSampleRateHeader(VDIFBaseHeader):
    """Base for VDIF headers that include the sample rate (EDV= 1, 3, 4)."""
    _header_parser = VDIFBaseHeader._header_parser + HeaderParser(
        (('sampling_unit', (4, 23, 1)),
         ('sampling_rate', (4, 0, 23)),
         ('sync_pattern', (5, 0, 32, 0xACABFEED))))

    # Add extra properties, ensuring 'time' comes after 'framerate', since
    # time setting requires the frame rate.
    _properties = (VDIFBaseHeader._properties[:-1] +
                   ('bandwidth', 'framerate', 'time'))

    def same_stream(self, other):
        return (super(VDIFSampleRateHeader, self).same_stream(other) and
                self.words[4] == other.words[4] and
                self.words[5] == other.words[5])

    @property
    def bandwidth(self):
        return u.Quantity(self['sampling_rate'],
                          u.MHz if self['sampling_unit'] else u.kHz)

    @bandwidth.setter
    def bandwidth(self, bandwidth):
        self['sampling_unit'] = not (bandwidth.unit == u.kHz or
                                     bandwidth.to(u.MHz).value % 1 != 0)
        if self['sampling_unit']:
            self['sampling_rate'] = int(bandwidth.to(u.MHz).value)
        else:
            assert bandwidth.to(u.kHz).value % 1 == 0
            self['sampling_rate'] = int(bandwidth.to(u.kHz).value)

    @property
    def framerate(self):
        # Could use self.bandwidth here, but speed up the calculation by
        # changing to a Quantity only at the end.
        return u.Quantity(self['sampling_rate'] *
                          (1000000 if self['sampling_unit'] else 1000) /
                          (self.nchan * self.samples_per_frame) *
                          (1 if self['complex_data'] else 2), u.Hz)

    @framerate.setter
    def framerate(self, framerate):
        framerate = framerate.to(u.Hz)
        assert framerate.value % 1 == 0
        self.bandwidth = (framerate * self.samples_per_frame * self.nchan /
                          (1 if self['complex_data'] else 2))


class VDIFHeader1(VDIFSampleRateHeader):
    """VDIF Header for EDV=1.

    See http://www.vlbi.org/vdif/docs/vdif_extension_0x01.pdf
    """
    _edv = 1
    _header_parser = VDIFSampleRateHeader._header_parser + HeaderParser(
        (('das_id', (6, 0, 64, 0x0)),))


class VDIFHeader3(VDIFSampleRateHeader):
    """VDIF Header for EDV=3.

    See http://www.vlbi.org/vdif/docs/vdif_extension_0x03.pdf
    """
    _edv = 3
    _header_parser = VDIFSampleRateHeader._header_parser + HeaderParser(
        (('frame_length', (2, 0, 24, 629)),  # Repeat, to set default.
         ('loif_tuning', (6, 0, 32, 0x0)),
         ('_7_28_4', (7, 28, 4, 0x0)),
         ('dbe_unit', (7, 24, 4, 0x0)),
         ('if_nr', (7, 20, 4, 0x0)),
         ('subband', (7, 17, 3, 0x0)),
         ('sideband', (7, 16, 1, False)),
         ('major_rev', (7, 12, 4, 0x0)),
         ('minor_rev', (7, 8, 4, 0x0)),
         ('personality', (7, 0, 8))))

    def verify(self):
        super(VDIFHeader3, self).verify()
        assert self['frame_length'] == 629


class VDIFHeader2(VDIFBaseHeader):
    """VDIF Header for EDV=2.

    See http://www.vlbi.org/vdif/docs/alma-vdif-edv.pdf

    Notes
    -----
    This header is untested.  It may need to have subclasses, based on possible
    differentsync values.
    """
    _edv = 2
    _header_parser = VDIFBaseHeader._header_parser + HeaderParser(
        (('complex_data', (3, 31, 1, 0x0)),  # Repeat, to set default.
         ('bits_per_sample', (3, 26, 5, 0x1)),  # Repeat, to set default.
         ('pol', (4, 0, 1)),
         ('BL_quadrant', (4, 1, 2)),
         ('BL_correlator', (4, 3, 1)),
         ('sync_pattern', (4, 4, 20, 0xa5ea5)),
         ('PIC_status', (5, 0, 32)),
         ('PSN', (6, 0, 64))))

    def verify(self):  # pragma: no cover
        super(VDIFHeader2, self).verify()
        assert self['frame_length'] == 629 or self['frame_length'] == 1004
        assert self.bps == 2 and not self['complex_data']


class VDIFMark5BHeader(VDIFBaseHeader, Mark5BHeader):
    """Mark 5B over VDIF (EDV=0xab).

    See http://www.vlbi.org/vdif/docs/vdif_extension_0xab.pdf
    """
    _edv = 0xab
    # Repeat 'frame_length' to set default.
    _header_parser = (VDIFBaseHeader._header_parser +
                      HeaderParser((('frame_length', (2, 0, 24, 1254)),)) +
                      HeaderParser(tuple(
                          ((k if k != 'frame_nr' else 'mark5b_frame_nr'),
                           (v[0]+4,) + v[1:])
                          for (k, v) in Mark5BHeader._header_parser.items())))

    def verify(self):
        super(VDIFMark5BHeader, self).verify()
        assert self['frame_length'] == 1254  # payload+header=10000+32 bytes/8
        assert self['frame_nr'] == self['mark5b_frame_nr']
        # check consistency of time down to the second (since some Mark 5B
        # headers do not store 'bcd_fraction').
        day, seconds = divmod(self['seconds'], 86400)
        assert seconds == self.seconds  # Latter decodes 'bcd_seconds'
        ref_mjd = ref_epochs[self['ref_epoch']].mjd + day
        assert ref_mjd % 1000 == self.jday  # Latter decodes 'bcd_jday'

    def __setitem__(self, item, value):
        super(VDIFMark5BHeader, self).__setitem__(item, value)
        if item == 'frame_nr':
            super(VDIFMark5BHeader, self).__setitem__('mark5b_frame_nr', value)

    def get_time(self, framerate=None, frame_nr=None):
        """
        Convert ref_epoch, seconds, and fractional seconds to Time object.

        Uses 'ref_epoch', which stores the number of half-years from 2000,
        and 'seconds', from the VDIF part of the header, and the fractional
        seconds from the Mark 5B part.

        Since some Mark 5B headers do not store the fractional seconds,
        one can also calculates the offset using the current frame number by
        passing in a frame rate.

        Furthermore, fractional seconds are stored only to 0.1 ms accuracy.
        In the code, this is "unrounded" to give the exact time of the start
        of the frame for any total bit rate below 512 Mbps.  For rates above
        this value, it is no longer guaranteed that subsequent frames have
        unique rates, and one should pass in an explicit frame rate instead.

        Set frame_nr=0 to just get the header time from ref_epoch and seconds.

        Parameters
        ----------
        framerate : `~astropy.units.Quantity`, optional
            For non-zero `frame_nr`, this is used to calculate the
            corresponding offset.
        frame_nr : int, optional
            Can be used to override the ``frame_nr`` from the header.  If 0,
            the routine simply returns the time to the integer second.

        Returns
        -------
        `~astropy.time.Time`
        """
        if framerate is None and frame_nr is None:
            # Get fractional second from the Mark 5B part of the header.
            offset = self.ns * 1.e-9
        else:
            if frame_nr is None:
                frame_nr = self['frame_nr']

            if frame_nr == 0:
                offset = 0.
            else:
                if framerate is None:
                    raise ValueError("calculating the time for a non-zero "
                                     "frame number requires a frame rate. "
                                     "Pass it in explicitly.")
                offset = (frame_nr / framerate).to(u.s).value

        return (ref_epochs[self['ref_epoch']] +
                TimeDelta(self['seconds'], offset, format='sec', scale='tai'))

    def set_time(self, time):
        Mark5BHeader.set_time(self, time)
        super(VDIFMark5BHeader, self).set_time(time, frame_nr=self['frame_nr'])

    time = property(get_time, set_time)
