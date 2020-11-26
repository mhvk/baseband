# Licensed under the GPLv3 - see LICENSE
"""
Definitions for VLBI VDIF Headers.

Implements a VDIFHeader class used to store header words, and decode/encode
the information therein.

For the VDIF specification, see https://www.vlbi.org/vdif
"""

import numpy as np
import astropy.units as u
from astropy.time import Time, TimeDelta

from ..base.header import (four_word_struct, eight_word_struct,
                           HeaderParser, VLBIHeaderBase)
from ..base.utils import fixedvalue
from ..mark5b.header import Mark5BHeader


__all__ = ['VDIFHeader', 'VDIFBaseHeader',
           'VDIFNoSampleRateHeader', 'VDIFSampleRateHeader',
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
    `~baseband.vdif.VDIFHeader`. This feature can be used to insert new EDV
    types into ``VDIF_HEADER_CLASSES``; see the :ref:`New VDIF EDV <new_edv>`
    tutorial.
    """
    _registry = VDIF_HEADER_CLASSES

    def __init__(cls, name, bases, dct):

        # Ignore VDIFHeader and VDIFBaseHeader, register others
        if name not in ('VDIFHeader', 'VDIFBaseHeader',
                        'VDIFNoSampleRateHeader', 'VDIFSampleRateHeader'):

            # Extract edv from class; convert to -1 if edv is False
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

            VDIFHeaderMeta._registry[edv] = cls

        # If header parser has a sync pattern, append it as a private
        # attribute for cls.verify.
        if (hasattr(cls, '_header_parser')
                and 'sync_pattern' in cls._header_parser.keys()):
            cls._sync_pattern = cls._header_parser.defaults['sync_pattern']

        super().__init__(name, bases, dct)


class VDIFHeader(VLBIHeaderBase, metaclass=VDIFHeaderMeta):
    """VDIF Header, supporting different Extended Data Versions.

    Will initialize a header instance appropriate for a given EDV.  See
    https://vlbi.org/wp-content/uploads/2019/03/VDIF_specification_Release_1.1.1.pdf

    Parameters
    ----------
    words : tuple of int, or None
        Eight (or four for legacy VDIF) 32-bit unsigned int header words.
        If `None`, set to a tuple of zeros for later initialisation.
    edv : int, False, or None, optional
        Extended data version.  If `False`, a legacy header is used.
        If `None` (default), it is determined from the header words;
        hence, setting it explicitly is useful mostly for a slight speed-up.
        (Subclasses can override this default with an ``_edv`` attribute.)
    verify : bool
        Whether to do basic verification of integrity.  Default: `True`.

    Returns
    -------
    header : `~baseband.vdif.VDIFHeader` subclass
        As appropriate for the extended data version.
    """

    _invariants = set()
    """Keys of invariant parts in all VDIF headers (none)."""
    _stream_invariants = {'legacy_mode', 'vdif_version', 'lg2_nchan',
                          'frame_length', 'complex_data', 'bits_per_sample',
                          'station_id'}
    """Keys of invariant parts in a given VDIF stream."""
    # One would think one could add 'ref_epoch' here, but timestamps
    # are broken for some threads of the EVN 2014 data, so apparently
    # that is not a safe bet.

    _properties = ('frame_nbytes', 'payload_nbytes', 'bps', 'complex_data',
                   'nchan', 'sample_shape', 'samples_per_frame',
                   'station', 'ref_time', 'time')
    """Properties accessible/usable in initialisation for all VDIF headers."""

    _edv = None
    _struct = eight_word_struct

    def __new__(cls, words=None, edv=None, verify=True, **kwargs):
        # We use edv to define which class we return.
        if edv is None:
            # If not given, try the class default.
            edv = cls._edv
            # If that is not given either, try to extract edv from the header
            # words, using parsers defined below, in VDIFBaseHeader.
            if edv is None and words is not None:
                base_parsers = VDIFBaseHeader._header_parser.parsers
                if base_parsers['legacy_mode'](words):
                    edv = False
                else:
                    edv = base_parsers['edv'](words)

        # Have to use key "-1" instead of "False" since the dict-lookup treats
        # 0 and False as identical.
        cls = VDIF_HEADER_CLASSES.get(edv if edv is not False else -1,
                                      VDIFBaseHeader)
        return super().__new__(cls)

    def __init__(self, words, edv=None, verify=True, **kwargs):
        if edv is not None:
            self._edv = edv
        super().__init__(words, verify=verify, **kwargs)

    def copy(self):
        return super().copy(edv=self.edv)

    def same_stream(self, other):
        """Whether header is consistent with being from the same stream."""
        return all(self[key] == other[key] for key in self.invariants())

    @classmethod
    def fromfile(cls, fh, edv=None, verify=True):
        """Read VDIF Header from file.

        Parameters
        ----------
        fh : filehandle
            To read data from.
        edv : int, False, or None, optional
            Extended data version.  If `False`, a legacy header is used.
            If `None` (default), it is determined from the header.  (Given it
            explicitly is mostly useful for a slight speed-up.)
        verify : bool, optional
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
    def fromvalues(cls, edv=False, *, verify=True, **kwargs):
        """Initialise a header from parsed values.

        Here, the parsed values must be given as keyword arguments, i.e., for
        any ``header = cls(<data>)``, ``cls.fromvalues(**header) == header``.

        However, unlike for the :meth:`~baseband.vdif.VDIFHeader.fromkeys`
        class method, data can also be set using arguments named after methods,
        such as ``bps`` and ``time``.

        Given defaults:

        invalid_data : `False`
        legacy_mode : `False`
        vdif_version : 1
        thread_id : 0
        frame_nr : 0
        sync_pattern : 0xACABFEED for EDV 1 and 3, 0xa5ea5 for EDV 2

        Values set by other keyword arguments (if present):

        bits_per_sample : from ``bps``
        frame_length : from ``samples_per_frame`` or ``frame_nbytes``
        lg2_nchan : from ``nchan``
        station_id : from ``station``
        sampling_rate, sampling_unit : from ``sample_rate``
        ref_epoch, seconds, frame_nr : from ``time``

        Note that to set ``time`` to non-integer seconds one also needs to
        pass in ``frame_rate`` or ``sample_rate``.
        """
        # Some defaults that are not done by setting properties.
        kwargs.setdefault('legacy_mode', edv is False)
        if edv is not False:
            kwargs['edv'] = edv

        if 'time' in kwargs and 'ref_epoch' not in kwargs:
            kwargs.setdefault('ref_time', kwargs['time'])

        return super().fromvalues(edv, verify=verify, **kwargs)

    @classmethod
    def fromkeys(cls, **kwargs):
        """Initialise a header from parsed values.

        Like :meth:`~baseband.vdif.VDIFHeader.fromvalues`, but without any
        interpretation of keywords.

        Raises
        ------
        KeyError : if not all keys required are pass in.
        """
        # Get all required values.
        edv = False if kwargs['legacy_mode'] else kwargs['edv']
        return super().fromkeys(edv, **kwargs)

    @classmethod
    def from_mark5b_header(cls, mark5b_header, bps, nchan, **kwargs):
        """Construct an Mark5B over VDIF header (EDV=0xab).

        See https://vlbi.org/wp-content/uploads/2019/03/vdif_extension_0xab.pdf

        Note that the Mark 5B header does not encode the bits-per-sample and
        the number of channels used in the payload, so these need to be given
        separately.  A complete frame can be encapsulated with
        `~baseband.vdif.VDIFFrame.from_mark5b_frame`.

        Parameters
        ----------
        mark5b_header : `~baseband.mark5b.Mark5BHeader`
            Used to set time, etc.
        bps : int
            Bits per elementary sample.
        nchan : int
            Number of channels carried in the Mark 5B payload.
        **kwargs
            Any further arguments.  Strictly, none are necessary to create a
            valid VDIF header, but this can be used to pass on, e.g.,
            ``invalid_data``.
        """
        assert 'time' not in kwargs, "Time is inferred from Mark 5B Header."
        # We need to treat the time carefully since the Mark 5B time may miss
        # fractional seconds, or they may be incorrect for rates above 512Mbps.
        # Integer seconds and frame_nr define everything uniquely, so we get
        # the time for frame number 0 and then set the frame number after.
        kwargs.update(mark5b_header)
        # Get time to integer seconds from mark5b header.
        time_frame0 = Time(mark5b_header.kday + mark5b_header.jday,
                           mark5b_header.seconds / 86400,
                           format='mjd', scale='utc')
        self = cls.fromvalues(edv=0xab, bps=bps, nchan=nchan,
                              complex_data=False, time=time_frame0, **kwargs)
        # Time setting will set frame_nr and bcd_fraction to 0; reset these
        # to the correct values.
        self['frame_nr'] = mark5b_header['frame_nr']
        self['bcd_fraction'] = mark5b_header['bcd_fraction']
        return self

    # Properties common to all VDIF headers.
    @property
    def edv(self):
        """VDIF Extended Data Version (EDV)."""
        return self._edv

    @property
    def frame_nbytes(self):
        """Size of the frame in bytes."""
        return self['frame_length'] * 8

    @frame_nbytes.setter
    def frame_nbytes(self, nbytes):
        assert nbytes % 8 == 0
        self['frame_length'] = int(nbytes) // 8

    @property
    def payload_nbytes(self):
        """Size of the payload in bytes."""
        return self.frame_nbytes - self.nbytes

    @payload_nbytes.setter
    def payload_nbytes(self, nbytes):
        self.frame_nbytes = nbytes + self.nbytes

    @property
    def bps(self):
        """Bits per elementary sample."""
        return self['bits_per_sample'] + 1

    @bps.setter
    def bps(self, bps):
        assert bps % 1 == 0
        int_bps = int(bps)
        if (int_bps & (int_bps - 1)) and self.nchan != 1:
            raise ValueError("bits per sample that is not a power of two "
                             "is only possible for single-channel data.")
        self['bits_per_sample'] = int(bps) - 1

    @property
    def complex_data(self):
        """Whether the data are complex."""
        return self['complex_data']

    @complex_data.setter
    def complex_data(self, complex_data):
        self['complex_data'] = complex_data

    @property
    def nchan(self):
        """Number of channels in the frame."""
        return 2 ** self['lg2_nchan']

    @nchan.setter
    def nchan(self, nchan):
        lg2_nchan = np.log2(nchan)
        assert lg2_nchan % 1 == 0
        if nchan != 1 and (self.bps & (self.bps - 1)) != 0:
            raise ValueError("Multi-channel data requires bits per sample "
                             "that is a power of two.")
        self['lg2_nchan'] = int(lg2_nchan)

    @property
    def sample_shape(self):
        """Shape of a sample in the payload (nchan,)."""
        return (self.nchan,)

    @sample_shape.setter
    def sample_shape(self, sample_shape):
        self.nchan, = sample_shape

    @property
    def samples_per_frame(self):
        """Number of complete samples in the frame."""
        # Values are not split over word boundaries.
        values_per_word = 32 // self.bps // (2 if self['complex_data'] else 1)
        # samples are not split over payload boundaries.
        return self.payload_nbytes // 4 * values_per_word // self.nchan

    @samples_per_frame.setter
    def samples_per_frame(self, samples_per_frame):
        values_per_word = 32 // self.bps // (2 if self['complex_data'] else 1)
        # units of frame length are 8 bytes, i.e., 2 words.
        values_per_long = values_per_word * 2
        longs = (samples_per_frame * self.nchan - 1) // values_per_long + 1
        old_payload_nbytes = self.payload_nbytes
        self.payload_nbytes = int(8 * longs)
        # Check that samples_per_frame would be recovered.
        if self.samples_per_frame != samples_per_frame:
            exc = ValueError("header cannot store {} samples per frame. "
                             "Nearest is {}.".format(samples_per_frame,
                                                     self.samples_per_frame))
            self.payload_nbytes = old_payload_nbytes
            raise exc

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

    @property
    def ref_time(self):
        """Reference time for the header, as encoded by 'ref_epoch'.

        If set, will use the nearest reference epoch before the given time.
        """
        return ref_epochs[self['ref_epoch']]

    @ref_time.setter
    def ref_time(self, ref_time):
        assert ref_time > ref_epochs[0]
        ref_index = np.searchsorted((ref_epochs - ref_time).sec, 0) - 1
        self['ref_epoch'] = ref_index

    def get_time(self, frame_rate=None):
        """Converts ref_epoch, seconds, and frame_nr to Time object.

        Uses 'ref_epoch', which stores the number of half-years from 2000,
        and 'seconds'.  By default, it also calculates the offset using
        the current frame number.  For non-zero 'frame_nr', this requires the
        frame rate, which is calculated from the sample rate in the header.

        Parameters
        ----------
        frame_rate : `~astropy.units.Quantity`, optional
            For non-zero 'frame_nr', this is required to calculate the
            corresponding offset.

        Returns
        -------
        time : `~astropy.time.Time`
        """
        frame_nr = self['frame_nr']
        if frame_nr == 0:
            offset = 0.
        else:
            if frame_rate is None:
                raise ValueError("this header does not provide a frame "
                                 "rate. Pass it in explicitly.")

            offset = (frame_nr / frame_rate).to_value(u.s)

        return (self.ref_time
                + TimeDelta(self['seconds'], offset, format='sec'))

    def set_time(self, time, frame_rate=None):
        """Converts Time object to ref_epoch, seconds, and frame_nr.

        For non-integer seconds, a frame rate is needed to calculate the
        'frame_nr'.

        Parameters
        ----------
        time : `~astropy.time.Time`
            The time to use for this header.
        frame_rate : `~astropy.units.Quantity`, optional
            For calculating 'frame_nr' from the fractional seconds.
        """
        seconds = time - self.ref_time
        int_sec = int(seconds.sec)

        # Round to nearest ns to handle timestamp difference errors.
        frac_sec = seconds - int_sec * u.s
        if abs(frac_sec) < 1. * u.ns:
            frame_nr = 0
        elif abs(1. * u.s - frac_sec) < 1. * u.ns:
            int_sec += 1
            frame_nr = 0
        else:
            if frame_rate is None:
                raise ValueError("this header does not provide a frame "
                                 "rate. Pass it in explicitly.")

            frame_nr = int((frac_sec * frame_rate).to(u.one).round())
            if abs(frame_nr / frame_rate - 1. * u.s) < 1. * u.ns:
                frame_nr = 0
                int_sec += 1

        self['seconds'] = int_sec
        self['frame_nr'] = frame_nr

    time = property(get_time, set_time)


class VDIFNoSampleRateHeader(VDIFHeader):
    def update(self, *, time=None, frame_rate=None, sample_rate=None,
               verify=True, **kwargs):
        """Update the header by setting keywords or properties.

        Here, any keywords matching header keys are applied first, and any
        remaining ones are used to set header properties, in the order set
        by the class (in ``_properties``).

        Parameters
        ----------
        time : `~astropy.time.Time`, optional
            A possible new time.  This is updated last.
        frame_rate : `~astropy.units.Quantity`, optional
            The frame rate to use in calculating the frame number from the
            time.  Needed for times at non-integer seconds.  Ignored if no
            ``time`` is given.
        sample_rate : `~astropy.units.Quantity`, optional
            Alternative to ``frame_rate``.  Ignored if no ``time`` is given,
            or if ``frame_rate`` is given.
        verify : bool, optional
            If `True` (default), verify integrity after updating.
        **kwargs
            Arguments used to set keywords and properties.
        """
        super().update(verify=False, **kwargs)

        if time is not None:
            if frame_rate is None and sample_rate is not None:
                frame_rate = sample_rate / self.samples_per_frame

            self.set_time(time, frame_rate=frame_rate)

        if verify:
            self.verify()


class VDIFLegacyHeader(VDIFNoSampleRateHeader):
    """Legacy VDIF header that uses only 4 32-bit words.

    See Section 6 of
    https://vlbi.org/wp-content/uploads/2019/03/VDIF_specification_Release_1.1.1.pdf
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
         ('frame_length', (2, 0, 24, 0x80)),
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
        assert self['frame_length'] >= 2


class VDIFBaseHeader(VDIFHeader):
    """Base for non-legacy VDIF headers that use 8 32-bit words."""

    _header_parser = VDIFLegacyHeader._header_parser | HeaderParser(
        (('legacy_mode', (0, 30, 1, False)),  # Repeat, to change default.
         ('edv', (4, 24, 8))))

    _invariants = (VDIFHeader._invariants
                   | {'legacy_mode'})
    """Keys of invariant parts in all VDIF non-legacy headers."""
    _stream_invariants = (_invariants
                          | VDIFHeader._stream_invariants
                          | {'edv'})
    """Keys of invariant parts in a given VDIF non-legacy stream."""

    def verify(self):
        """Basic checks of header integrity."""
        assert not self['legacy_mode']
        assert self.edv is None or self.edv == self['edv']
        assert len(self.words) == 8
        assert self['frame_length'] >= 4
        # _sync_pattern is added by VDIFHeaderMeta.
        if 'sync_pattern' in self.keys():
            assert self['sync_pattern'] == self._sync_pattern


class VDIFHeader0(VDIFBaseHeader, VDIFNoSampleRateHeader):
    """VDIF Header for EDV=0.

    EDV=0 implies the extended user data fields are not used.
    """
    _edv = 0

    def verify(self):
        super().verify()
        assert all(word == 0 for word in self.words[4:])


class VDIFSampleRateHeader(VDIFBaseHeader):
    """Base for VDIF headers that include the sample rate (EDV= 1, 3, 4)."""

    _header_parser = VDIFBaseHeader._header_parser | HeaderParser(
        (('sampling_unit', (4, 23, 1)),
         ('sampling_rate', (4, 0, 23)),
         ('sync_pattern', (5, 0, 32, 0xACABFEED))))

    _invariants = (VDIFBaseHeader._invariants
                   | {'sync_pattern'})
    """Keys of invariant parts in VDIF headers with sample rate."""
    _stream_invariants = (_invariants
                          | VDIFBaseHeader._stream_invariants
                          | {'sampling_unit', 'sampling_rate'})
    """Keys of invariant parts in a given VDIF sample-rate stream."""

    # Add extra properties, ensuring 'time' comes after 'sample_rate' and
    # 'frame_rate', since time setting requires the frame rate.
    _properties = (VDIFBaseHeader._properties[:-1]
                   + ('sample_rate', 'frame_rate', 'time'))

    @property
    def sample_rate(self):
        """Number of complete samples per second.

        Assumes the 'sampling_rate' header field represents a per-channel
        sample rate for complex samples, or half the sample rate for real ones.
        """
        # Interprets sample rate correctly for EDV=3, but may not for EDV=1.
        return u.Quantity(self['sampling_rate']
                          * (1 if self['complex_data'] else 2),
                          u.MHz if self['sampling_unit'] else u.kHz)

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        # Sets sample rate correctly for EDV=3, but may not for EDV=1 or for
        # multiple channels per frame (illegal for EDV=3).
        assert sample_rate.to_value(u.Hz) % 1 == 0
        complex_sample_rate = sample_rate / (1 if self['complex_data'] else 2)
        self['sampling_unit'] = not (
            complex_sample_rate.unit == u.kHz
            or complex_sample_rate.to_value(u.MHz) % 1 != 0)
        if self['sampling_unit']:
            self['sampling_rate'] = int(complex_sample_rate.to_value(u.MHz))
        else:
            assert complex_sample_rate.to(u.kHz).value % 1 == 0
            self['sampling_rate'] = int(complex_sample_rate.to_value(u.kHz))

    @property
    def frame_rate(self):
        """Number of frames per second.

        Assumes the 'sampling_rate' header field represents a per-channel
        sample rate for complex samples, or half the sample rate for real ones.
        """
        return self.sample_rate / self.samples_per_frame

    @frame_rate.setter
    def frame_rate(self, frame_rate):
        self.sample_rate = frame_rate * self.samples_per_frame

    def get_time(self, frame_rate=None):
        """Converts ref_epoch, seconds, and frame_nr to Time object.

        Uses 'ref_epoch', which stores the number of half-years from 2000,
        and 'seconds'.  By default, it also calculates the offset using
        the current frame number.  For non-zero 'frame_nr', this requires the
        frame rate, which is calculated from the sample rate in the header.
        The latter can also be explicitly passed on.

        Parameters
        ----------
        frame_rate : `~astropy.units.Quantity`, optional
            For non-zero 'frame_nr', this is used to calculate the
            corresponding offset. If not given, the frame rate from the
            header is used (if it is non-zero).

        Returns
        -------
        time : `~astropy.time.Time`
        """
        if frame_rate is None and self['sampling_rate'] != 0:
            frame_rate = self.frame_rate
        return super().get_time(frame_rate=frame_rate)

    def set_time(self, time, frame_rate=None):
        """Converts Time object to ref_epoch, seconds, and frame_nr.

        Parameters
        ----------
        time : `~astropy.time.Time`
            The time to use for this header.
        frame_rate : `~astropy.units.Quantity`, optional
            For calculating 'frame_nr' from the fractional seconds. If not
            given, the frame rate from the header is used (if it is non-zero).
        """
        if frame_rate is None and self['sampling_rate'] != 0:
            frame_rate = self.frame_rate
        super().set_time(time, frame_rate=frame_rate)

    time = property(get_time, set_time)


class VDIFHeader1(VDIFSampleRateHeader):
    """VDIF Header for EDV=1.

    See https://vlbi.org/wp-content/uploads/2019/03/vdif_extension_0x01.pdf
    """
    _edv = 1
    _header_parser = VDIFSampleRateHeader._header_parser | HeaderParser(
        (('das_id', (6, 0, 64, 0x0)),))

    _invariants = (VDIFSampleRateHeader._invariants
                   | {'edv'})
    """Keys of invariant parts in all VDIF EDV=1 headers."""


class VDIFHeader3(VDIFSampleRateHeader):
    """VDIF Header for EDV=3.

    See https://vlbi.org/wp-content/uploads/2019/03/vdif_extension_0x03.pdf
    """
    _edv = 3
    _header_parser = VDIFSampleRateHeader._header_parser | HeaderParser(
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

    _invariants = (VDIFSampleRateHeader._invariants
                   | {'edv', 'frame_length'})
    """Keys of invariant parts in all VDIF EDV=3 headers."""
    _stream_invariants = (_invariants
                          | VDIFSampleRateHeader._stream_invariants
                          | {'major_rev', 'minor_rev', 'personality'})
    """Keys of invariant parts in a given VDIF EDV=3 stream."""

    def verify(self):
        super().verify()
        assert self['frame_length'] in (129, 629)

    @property
    def frame_nbytes(self):
        """Size of the frame in bytes."""
        return self['frame_length'] * 8

    @frame_nbytes.setter
    def frame_nbytes(self, nbytes):
        assert nbytes in (1032, 5032)
        self['frame_length'] = int(nbytes) // 8


class VDIFHeader2(VDIFBaseHeader, VDIFNoSampleRateHeader):
    """VDIF Header for EDV=2.

    See https://vlbi.org/wp-content/uploads/2019/03/alma-vdif-edv.pdf

    Notes
    -----
    This header is untested.  It may need to have subclasses, based on possible
    different sync values.
    """
    _edv = 2

    _header_parser = VDIFBaseHeader._header_parser | HeaderParser(
        (('complex_data', (3, 31, 1, 0x0)),  # Repeat, to set default.
         ('bits_per_sample', (3, 26, 5, 0x1)),  # Repeat, to set default.
         ('pol', (4, 0, 1)),
         ('BL_quadrant', (4, 1, 2)),
         ('BL_correlator', (4, 3, 1)),
         ('sync_pattern', (4, 4, 20, 0xa5ea5)),
         ('PIC_status', (5, 0, 32)),
         ('PSN', (6, 0, 64))))

    _invariants = (VDIFBaseHeader._invariants
                   | {'edv', 'sync_pattern'})
    """Keys of invariant parts in all VDIF EDV=2 headers."""
    _stream_invariants = (_invariants
                          | VDIFBaseHeader._stream_invariants)
    """Keys of invariant parts in a given VDIF EDV=2 stream."""

    def verify(self):  # pragma: no cover
        super().verify()
        assert self['frame_length'] == 629 or self['frame_length'] == 1004
        assert self.bps == 2 and not self['complex_data']


class VDIFMark5BHeader(VDIFBaseHeader, VDIFNoSampleRateHeader, Mark5BHeader):
    """Mark 5B over VDIF (EDV=0xab).

    See https://vlbi.org/wp-content/uploads/2019/03/vdif_extension_0xab.pdf
    """
    _edv = 0xab
    # Repeat 'frame_length' to set default.
    _header_parser = (VDIFBaseHeader._header_parser
                      | HeaderParser((('frame_length', (2, 0, 24, 1254)),))
                      | HeaderParser(tuple(
                          ((k if k != 'frame_nr' else 'mark5b_frame_nr'),
                           (v[0] + 4,) + v[1:])
                          for (k, v) in Mark5BHeader._header_parser.items())))

    _invariants = (VDIFBaseHeader._invariants
                   | {'frame_length'}
                   | Mark5BHeader._invariants)
    """Keys of invariant parts in all VDIF-Mark5B headers."""
    _stream_invariants = (_invariants
                          | VDIFBaseHeader._stream_invariants
                          | Mark5BHeader._stream_invariants)
    """Keys of invariant parts in a given VDIF-Mark5B stream."""

    def verify(self):
        super().verify()
        assert self['frame_length'] == 1254  # payload+header=10000+32 bytes/8
        assert self['frame_nr'] == self['mark5b_frame_nr']
        assert not self['complex_data']
        # Check consistency of time down to the second (since some Mark 5B
        # headers do not store 'bcd_fraction').
        day, seconds = divmod(self['seconds'], 86400)
        assert seconds == self.seconds  # Latter decodes 'bcd_seconds'
        ref_mjd = ref_epochs[self['ref_epoch']].mjd + day
        assert ref_mjd % 1000 == self.jday  # Latter decodes 'bcd_jday'

    @fixedvalue
    def payload_nbytes(cls):
        return Mark5BHeader.payload_nbytes

    @fixedvalue
    def frame_nbytes(cls):
        return cls.nbytes + cls.payload_nbytes

    @fixedvalue
    def complex_data(cls):
        return False

    def __setitem__(self, item, value):
        if item == 'complex_data':
            # Pass by fixed-value setter - can only set to False.
            self.complex_data = value
        else:
            super().__setitem__(item, value)
            if item == 'frame_nr':
                super().__setitem__('mark5b_frame_nr', value)

    def get_time(self, frame_rate=None):
        """
        Convert ref_epoch, seconds, and fractional seconds to Time object.

        Uses 'ref_epoch', which stores the number of half-years from 2000,
        and 'seconds', from the VDIF part of the header, and the fractional
        seconds from the Mark 5B part.

        Since some Mark 5B headers do not store the fractional seconds,
        one can also calculates the offset using the current frame number by
        passing in a sample rate.

        Furthermore, fractional seconds are stored only to 0.1 ms accuracy.
        In the code, this is "unrounded" to give the exact time of the start
        of the frame for any total bit rate below 512 Mbps.  For rates above
        this value, it is no longer guaranteed that subsequent frames have
        unique rates, and one should pass in an explicit sample rate instead.

        Parameters
        ----------
        frame_rate : `~astropy.units.Quantity`, optional
            For non-zero 'frame_nr', this is used to calculate the
            corresponding offset.

        Returns
        -------
        time : `~astropy.time.Time`
        """
        frame_nr = self['frame_nr']
        if frame_nr == 0:
            fraction = 0.
        elif frame_rate is None:
            # Get fractional second from the Mark 5B part of the header,
            # but check it is non-zero (it doesn't always seem to be set).
            fraction = self.fraction
            if fraction == 0.:
                raise ValueError('header does not provide correct fractional '
                                 'second (it is zero for non-zero frame '
                                 'number). Please pass in a frame_rate.')
        else:
            fraction = (frame_nr / frame_rate).to_value(u.s)

        return (ref_epochs[self['ref_epoch']]
                + TimeDelta(self['seconds'], fraction,
                            format='sec', scale='tai'))

    def set_time(self, time, frame_rate=None):
        Mark5BHeader.set_time(self, time, frame_rate)
        super().set_time(time, frame_rate)

    time = property(get_time, set_time)
