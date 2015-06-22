from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import warnings

import numpy as np
import astropy.units as u

from astropy.time import Time, TimeDelta

from ..vlbi_helpers import HeaderParser, four_word_struct, eight_word_struct


legacy_parser = HeaderParser(
    (('invalid_data', (0, 31, 1, False)),
     ('legacy_mode', (0, 30, 1, True)),
     ('seconds', (0, 0, 30)),
     ('ref_epoch', (1, 24, 6)),
     ('frame_nr', (1, 0, 24, 0x0)),
     ('_2_30_2', (2, 30, 2, 0x0)),
     ('vdif_version', (2, 29, 3, 0x1)),
     ('lg2_nchan', (2, 24, 5)),
     ('frame_length', (2, 0, 24)),
     ('complex_data', (3, 31, 1)),
     ('bits_per_sample', (3, 26, 5)),
     ('thread_id', (3, 16, 10, 0x0)),
     ('station_id', (3, 0, 16))))

base_parser = legacy_parser + HeaderParser(
    (('legacy_mode', (0, 30, 1, True)),  # Repeat, to change default.
     ('edv', (4, 24, 8))))

header_parsers = {
    False: legacy_parser,
    1: base_parser + HeaderParser(
        (('sampling_unit', (4, 23, 1)),
         ('sample_rate', (4, 0, 23)),
         ('sync_pattern', (5, 0, 32, 0xACABFEED)),
         ('das_id', (6, 0, 32, 0x0)),
         ('_7_0_32', (7, 0, 32, 0x0)))),
    3: base_parser + HeaderParser(
        (('frame_length', (2, 0, 24, 629)),  # Repeat, to set default.
         ('sampling_unit', (4, 23, 1)),
         ('sample_rate', (4, 0, 23)),
         ('sync_pattern', (5, 0, 32, 0xACABFEED)),
         ('loif_tuning', (6, 0, 32, 0x0)),
         ('_7_28_4', (7, 28, 4, 0x0)),
         ('dbe_unit', (7, 24, 4, 0x0)),
         ('if_nr', (7, 20, 4, 0x0)),
         ('subband', (7, 17, 3, 0x0)),
         ('sideband', (7, 16, 1, False)),
         ('major_rev', (7, 12, 4, 0x0)),
         ('minor_rev', (7, 8, 4, 0x0)),
         ('personality', (7, 0, 8)))),
    4: base_parser + HeaderParser(
        (('sampling_unit', (4, 23, 1)),
         ('sample_rate', (4, 0, 23)),
         ('sync_pattern', (5, 0, 32))))}

# Also have mark5b over vdif (edv = 0xab)
# http://www.vlbi.org/vdif/docs/vdif_extension_0xab.pdf


ref_max = int(2. * (Time.now().jyear - 2000.)) + 1
ref_epochs = Time(['{y:04d}-{m:02d}-01'.format(y=2000 + ref // 2,
                                               m=1 if ref % 2 == 0 else 7)
                   for ref in range(ref_max)], format='isot', scale='utc',
                  precision=9)


class VDIFHeader(object):

    _properties = ('framesize', 'payloadsize', 'bps', 'nchan',
                   'samples_per_frame', 'station', 'bandwidth',
                   'framerate', 'time')

    def __init__(self, words, edv=None, verify=True):
        """Interpret a tuple of words as a VDIF Frame Header."""
        self.words = words
        if edv is None:
            if base_parser.parsers['legacy_mode'](words):
                self.edv = False
            else:
                self.edv = base_parser.parsers['edv'](words)
        else:
            self.edv = edv

        # If edv is not known, use base header so that the basic properties
        # can still be gotten.
        self._header_parser = header_parsers.get(self.edv, base_parser)
        if verify:
            self.verify()

    def verify(self):
        """Basic checks of header integrity."""
        if self.edv is False:
            assert self['legacy_mode']
            assert len(self.words) == 4
        else:
            assert not self['legacy_mode']
            assert self.edv == self['edv']
            assert len(self.words) == 8
            if 'sync_pattern' in self.keys():
                assert (self['sync_pattern'] ==
                        self._header_parser.defaults['sync_pattern'])
            elif self.edv == 3:
                assert self['frame_length'] == 629

    def copy(self):
        return self.__class__(self.words, self.edv, verify=False)

    def same_stream(self, other):
        """Whether header is consistent with being from the same stream."""
        # Words 2 and 3 should be invariant, and edv should be the same.
        if not (self.edv == other.edv and
                all(self[key] == other[key]
                    for key in ('ref_epoch', 'vdif_version', 'frame_length',
                                'complex_data', 'bits_per_sample',
                                'station_id'))):
            return False

        if self.edv:
            # For any edv, word 4 should be invariant.
            return self.words[4] == other.words[4]
        else:
            return True

    @classmethod
    def frombytes(cls, s, edv=None, verify=True):
        """Read VDIF Header from bytes."""
        try:
            return cls(eight_word_struct.unpack(s), edv, verify)
        except:
            if edv:
                raise
            else:
                return cls(four_word_struct.unpack(s), False, verify)

    def tobytes(self):
        if len(self.words) == 8:
            return eight_word_struct.pack(*self.words)
        else:
            return four_word_struct.pack(*self.words)

    @classmethod
    def fromfile(cls, fh, edv=None, verify=True):
        """Read VDIF Header from file."""
        # Assume non-legacy header to ensure those are done fastest.
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

    def tofile(self, fh):
        return fh.write(self.tobytes())

    @classmethod
    def fromvalues(cls, **kwargs):
        """Initialise a header from parsed values.

        Here, the parsed values must be given as keyword arguments, i.e.,
        for any header = cls(<somedata>), cls.fromvalues(**header) == header.

        However, unlike for the 'fromkeys' class method, data can also be set
        using arguments named after header methods such as 'bps' and 'time'.

        Given defaults for standard header keywords:

        invalid_data : `False`
        legacy_mode : `False`
        vdif_version : 1
        thread_id: 0
        frame_nr: 0

        Defaults inferred from other keyword arguments (if present):

        bits_per_sample : from 'bps'
        frame_length : from 'framesize' (or 'payloadsize' and 'legacy_mode')
        lg2_nchan : from 'nchan'

        Given defaults for edv 1 and 3:

        sync_pattern: 0xACABFEED (for edv = 1 and 3)

        Defaults inferred from other keyword arguments for all edv:

        station_id : from 'station'
        sample_rate, sample_unit : from 'bandwidth' or 'framerate'
        ref_epoch, seconds, frame_nr : from 'time'
        """
        # Make a temporary header in which we try setting keywords
        edv = kwargs.get('edv', False)
        if edv is False:
            assert kwargs.get('legacy_mode', True)
            kwargs['legacy_mode'] = True
            words = (0, 0, 0, 0)
        else:
            assert not kwargs.get('legacy_mode', False)
            kwargs['legacy_mode'] = False
            words = (0, 0, 0, 0, 0, 0, 0, 0)

        self = cls(words, edv, verify=False)
        # First set all keys to keyword arguments or defaults.
        for key in self.keys():
            if key in kwargs:
                self[key] = kwargs.pop(key)
            elif self._header_parser.defaults[key] is not None:
                self[key] = self._header_parser.defaults[key]

        # Next, use remaining keyword arguments to set properties.
        # Order may be important so use list:
        for key in self._properties:
            if key in kwargs:
                setattr(self, key, kwargs.pop(key))

        if kwargs:
            warnings.warn("Some keywords unused in header initialisation: {0}"
                          .format(kwargs))

        return self

    @classmethod
    def fromkeys(cls, **kwargs):
        """Like fromvalues, but without any interpretation of keywords."""
        # Get all required values.
        if kwargs['legacy_mode']:
            words = (0, 0, 0, 0)
            edv = False
        else:
            words = (0, 0, 0, 0, 0, 0, 0, 0)
            edv = kwargs['edv']

        header_parser = header_parsers[edv]
        for key in header_parser:
            words = header_parser.setters[key](words, kwargs.pop(key))

        if kwargs:
            warnings.warn("Some keywords unused in header initialisation: {0}"
                          .format(kwargs))

        return cls(words, edv)

    def __getitem__(self, item):
        try:
            return self._header_parser.parsers[item](self.words)
        except KeyError:
            raise KeyError("VDIF Frame Header does not contain {0}"
                           .format(item))

    def __setitem__(self, item, value):
        try:
            self.words = self._header_parser.setters[item](self.words, value)
        except KeyError:
            raise KeyError("VDIF Frame Header does not contain {0}"
                           .format(item))

    def keys(self):
        return self._header_parser.keys()

    def __eq__(self, other):
        return (type(self) is type(other) and
                list(self.words) == list(other.words))

    def __contains__(self, key):
        return key in self.keys()

    def __repr__(self):
        return ("<{0} {1}>".format(
            self.__class__.__name__, ",\n            ".join(
                ["{0}: {1}".format(k, (hex(self[k]) if k == 'sync_pattern' else
                                       self[k])) for k in self.keys()])))

    @property
    def size(self):
        return len(self.words) * 4

    @property
    def framesize(self):
        return self['frame_length'] * 8

    @framesize.setter
    def framesize(self, size):
        assert size % 8 == 0
        self['frame_length'] = int(size) // 8

    @property
    def payloadsize(self):
        return self.framesize - self.size

    @payloadsize.setter
    def payloadsize(self, size):
        self.framesize = size + self.size

    @property
    def bps(self):
        bps = self['bits_per_sample'] + 1
        if self['complex_data']:
            bps *= 2
        return bps

    @bps.setter
    def bps(self, bps):
        if self['complex_data']:
            bps /= 2
        assert bps % 1 == 0
        self['bits_per_sample'] = int(bps) - 1

    @property
    def nchan(self):
        return 2**self['lg2_nchan']

    @nchan.setter
    def nchan(self, nchan):
        lg2_nchan = np.log2(nchan)
        assert lg2_nchan % 1 == 0
        self['lg2_nchan'] = int(lg2_nchan)

    @property
    def samples_per_frame(self):
        # Values are not split over word boundaries.
        values_per_word = 32 // self.bps
        # samples are not split over payload boundaries.
        return self.payloadsize // 4 * values_per_word // self.nchan

    @samples_per_frame.setter
    def samples_per_frame(self, samples_per_frame):
        values_per_long = (32 // self.bps) * 2
        longs, extra = divmod(samples_per_frame * self.nchan,
                              values_per_long)
        if extra:
            longs += 1

        self['frame_length'] = int(longs) + self.size // 8

    @property
    def station(self):
        msb = self['station_id'] >> 8
        if 48 <= msb < 128:
            return chr(msb) + chr(self['station_id'] & 0xff)
        else:
            return self['station_id']

    @station.setter
    def station(self, station):
        try:
            station_id = ord(station[0]) << 8 + ord(station[1])
        except TypeError:
            station_id = station
        assert int(station_id) == station_id
        self['station_id'] = station_id

    @property
    def bandwidth(self):
        return u.Quantity(self['sample_rate'],
                          u.MHz if self['sampling_unit'] else u.kHz)

    @bandwidth.setter
    def bandwidth(self, bandwidth):
        self['sampling_unit'] = not (bandwidth.unit == u.kHz or
                                     bandwidth.to(u.MHz).value % 1 != 0)
        if self['sampling_unit']:
            self['sample_rate'] = bandwidth.to(u.MHz).value
        else:
            assert bandwidth.to(u.kHz).value % 1 == 0
            self['sample_rate'] = bandwidth.to(u.kHz).value

    @property
    def framerate(self):
        # Could use self.bandwidth here, but speed up the calculation by
        # changing to a Quantity only at the end.
        return u.Quantity(self['sample_rate'] *
                          (1000000 if self['sampling_unit'] else 1000) *
                          2 * self.nchan / self.samples_per_frame, u.Hz)

    @framerate.setter
    def framerate(self, framerate):
        framerate = framerate.to(u.Hz)
        assert framerate.value % 1 == 0
        self.bandwidth = framerate * self.samples_per_frame / (2 * self.nchan)

    def get_time(self, frame_nr=None, framerate=None):
        """
        Convert ref_epoch, seconds, and possibly frame_nr to Time object.

        Uses 'ref_epoch', which stores the number of half-years from 2000,
        and 'seconds'.  By default, it also calculates the offset using
        the current frame number.  For non-zero frame_nr, this requires the
        framerate, which is calculated from the header.  It can be passed on
        if this is not available (e.g., for a legacy VDIF header).

        Set frame_nr=0 to just get the header time from ref_epoch and seconds.
        """
        if frame_nr is None:
            frame_nr = self['frame_nr']

        if frame_nr == 0:
            offset = 0.
        else:
            if framerate is None:
                framerate = self.framerate
            offset = (frame_nr / framerate).to(u.s).value
        return (ref_epochs[self['ref_epoch']] +
                TimeDelta(self['seconds'], offset, format='sec', scale='tai'))

    def set_time(self, time):
        assert time > ref_epochs[0]
        ref_index = np.searchsorted((ref_epochs - time).sec, 0) - 1
        self['ref_epoch'] = ref_index
        seconds = (time - ref_epochs[ref_index]).to(u.s)
        int_sec, frac_sec = divmod(seconds, 1 * u.s)
        self['seconds'] = int(int_sec)
        if abs(frac_sec) < 1. * u.ns:
            self['frame_nr'] = 0
        else:
            self['frame_nr'] = (frac_sec / self.samples_per_frame *
                                self.bandwidth).to(u.one).value

    time = property(get_time, set_time)
