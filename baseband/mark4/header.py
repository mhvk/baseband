"""
Definitions for VLBI Mark 4 Headers.

Implements a Mark4Header class used to store header words, and decode/encode
the information therein.

For the specification, see
http://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from astropy.time import Time
from astropy.extern import six

from ..vlbi_base.header import HeaderParser, VLBIHeaderBase
from ..vlbi_base.utils import bcd_decode, bcd_encode, CRC

__all__ = ['stream2words', 'words2stream', 'Mark4TrackHeader', 'Mark4Header']


PAYLOADSIZE = 20000
"""Number of bits per track per frame."""

CRC12 = 0x180f
"""CRC polynomial used for Mark 4 Headers.

x^12 + x^11 + x^3 + x^2 + x + 1, i.e., 0x180f.
See page 4 of http://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf

This is also a 'standard' CRC-12 mentioned in
https://en.wikipedia.org/wiki/Cyclic_redundancy_check
"""
crc12 = CRC(CRC12)


def stream2words(stream, track=None):
    """Convert a stream of integers to uint32 header words.

    Parameters
    ----------
    stream : array of int
        For each int, every bit corresponds to a particular track.
    track : int, array, or None
        The track to extract.  If `None` (default), extract all tracks that
        the type of int in the stream can hold.
    """
    if track is None:
        track = np.arange(stream.dtype.itemsize * 8, dtype=stream.dtype)

    track_sel = ((stream.reshape(-1, 32, 1) >> track) & 1).astype(np.uint32)
    track_sel <<= np.arange(31, -1, -1, dtype=np.uint32).reshape(1, 32, 1)
    words = np.bitwise_or.reduce(track_sel, axis=1)
    return words.squeeze()


def words2stream(words):
    """Convert a set of uint32 header words to a stream of integers.

    Parameters
    ----------
    words : array of uint32

    Returns
    -------
    stream : array of int
        For each int, every bit corresponds to a particular track.
    """
    dtype = np.dtype('<u{:1d}'.format(words.shape[1] // 8))
    bit = np.arange(words.shape[1], dtype=dtype)
    track = np.arange(31, -1, -1, dtype=words.dtype).reshape(-1, 1)

    track_sel = ((words[:, np.newaxis, :] >> track) & 1).astype(dtype)
    track_sel <<= bit
    words = np.bitwise_or.reduce(track_sel, axis=2)
    return words.ravel()


class Mark4TrackHeader(VLBIHeaderBase):
    """Decoder/encoder of a Mark 4 Track Header.

    See http://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf

    Parameters
    ----------
    words : tuple of int, or None
        Five 32-bit unsigned int header words.  If ``None``, set to a list
        of zeros for later initialisation.
    decade : int, or None
        Decade the observations were taken (needed to remove ambiguity in the
        Mark 4 time stamp).
    verify : bool
        Whether to do basic verification of integrity.  Default: `True`.

    Returns
    -------
    header : Mark4TrackHeader instance.
    """

    _header_parser = HeaderParser(
        (('bcd_headstack1', (0, 0, 16)),
         ('bcd_headstack2', (0, 16, 16)),
         ('headstack_id', (1, 30, 2)),
         ('bcd_track_id', (1, 24, 6)),
         ('fan_out', (1, 22, 2)),
         ('magnitude_bit', (1, 21, 1)),
         ('lsb_output', (1, 20, 1)),
         ('converter_id', (1, 16, 4)),
         ('time_sync_error', (1, 15, 1, False)),
         ('internal_clock_error', (1, 14, 1, False)),
         ('processor_time_out_error', (1, 13, 1, False)),
         ('communication_error', (1, 12, 1, False)),
         ('_1_11_1', (1, 11, 1, False)),
         ('_1_10_1', (1, 10, 1, False)),
         ('track_roll_enabled', (1, 9, 1, False)),
         ('sequence_suspended', (1, 8, 1, False)),
         ('system_id', (1, 0, 8)),
         ('sync_pattern', (2, 0, 32, 0xffffffff)),
         ('bcd_unit_year', (3, 28, 4)),
         ('bcd_day', (3, 16, 12)),
         ('bcd_hour', (3, 8, 8)),
         ('bcd_minute', (3, 0, 8)),
         ('bcd_second', (4, 24, 8)),
         ('bcd_fraction', (4, 12, 12)),
         ('crc', (4, 0, 12))))

    _properties = ('decade', 'track_id', 'ms', 'time')
    """Properties accessible/usable in initialisation."""

    decade = None

    def __init__(self, words, decade=None, verify=True):
        if words is None:
            self.words = [0, 0, 0, 0, 0]
        else:
            self.words = words
        if decade is not None:
            self.decade = decade
        if verify:
            self.verify()

    def verify(self):
        """Verify header integrity."""
        assert len(self.words) == 5
        assert np.all(self['sync_pattern'] ==
                      self._header_parser.defaults['sync_pattern'])
        assert np.all(self['bcd_fraction'] & 0xf) % 5 != 4
        assert self.decade is not None and (1950 < self.decade < 3000)

    @property
    def track_id(self):
        return bcd_decode(self['bcd_track_id'])

    @track_id.setter
    def track_id(self, track_id):
        self['bcd_track_id'] = bcd_encode(track_id)

    @property
    def ms(self):
        """Fractional seconds (in ms; decoded from 'bcd_fraction')."""
        ms = bcd_decode(self['bcd_fraction'])
        # The last digit encodes a fraction -- see table 2 in
        # http://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf
        # 0: 0.00      5: 5.00
        # 1: 1.25      6: 6.25
        # 2: 2.50      7: 7.50
        # 3: 3.75      8: 8.75
        # 4: invalid   9: invalid
        last_digit = ms % 5
        return ms + last_digit * 0.25

    @ms.setter
    def ms(self, ms):
        if np.any(np.abs((ms / 1.25) - np.round(ms / 1.25)) > 1e-6):
            raise ValueError("{0} ms is not a multiple of 1.25 ms"
                             .format(ms))
        self['bcd_fraction'] = bcd_encode(np.floor(ms + 1e-6)
                                          .astype(np.int32))

    def get_time(self):
        """
        Convert BCD time code to Time object.

        Uses bcd-encoded 'unit_year', 'day', 'hour', 'minute', 'second' and
        'frac_sec', plus ``decade`` from the initialisation to calculate the
        time.  See http://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf
        """
        return Time('{decade:03d}{uy:1x}:{d:03x}:{h:02x}:{m:02x}:{s:08.5f}'
                    .format(decade=self.decade//10, uy=self['bcd_unit_year'],
                            d=self['bcd_day'], h=self['bcd_hour'],
                            m=self['bcd_minute'],
                            s=bcd_decode(self['bcd_second']) + self.ms/1000),
                    format='yday', scale='utc', precision=5)

    def set_time(self, time):
        old_precision = time.precision
        try:
            time.precision = 5
            yday = time.yday.split(':')
        finally:
            time.precision = old_precision
        # ms first since that checks precision.
        self.ms = (float(yday[4]) % 1) * 1000
        self.decade = int(yday[0][:3]) * 10
        self['bcd_unit_year'] = int(yday[0][3], base=16)
        self['bcd_day'] = int(yday[1], base=16)
        self['bcd_hour'] = int(yday[2], base=16)
        self['bcd_minute'] = int(yday[3], base=16)
        self['bcd_second'] = int(yday[4][:2], base=16)

    time = property(get_time, set_time)


class Mark4Header(Mark4TrackHeader):
    """Decoder/encoder of a Mark 4 Header, containing all streams.

    See http://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf

    Parameters
    ----------
    words : ndarray of int, or None
        Shape should be (5, number-of-tracks), and dtype np.uint32.  If `None`,
        ``ntrack`` should be given and words will be initialized to 0.
    decade : int, or None
        Decade the observations were taken (needed to remove ambiguity in the
        Mark 4 time stamp).
    verify : bool
        Whether to do basic verification of integrity.  Default: `True`.
    ntrack : None or int
        To help initialize ``words`` if needed.

    Returns
    -------
    header : Mark4Header instance.
    """

    _track_header = Mark4TrackHeader
    _properties = (Mark4TrackHeader._properties +
                   ('ntrack', 'framesize', 'payloadsize', 'fanout',
                    'samples_per_frame', 'bps', 'nchan'))
    _dtypes = {1: 'b',
               2: 'u1',
               4: 'u1',
               8: 'u1',
               16: '<u2',
               32: '<u4',
               64: '<u8'}

    def __init__(self, words, ntrack=None, decade=None, verify=True):
        if words is None:
            self.words = np.zeros((5, ntrack), dtype=np.uint32)
        else:
            self.words = words
        if decade is not None:
            self.decade = decade
        if verify:
            self.verify()

    def verify(self):
        super(Mark4Header, self).verify()
        assert set(self['fan_out']) == set(np.arange(self.fanout))
        # The following cannot be assumed to be true, it seems.
        # assert set(self['converter_id']) == set(np.arange(self.nchan))

    @classmethod
    def _stream_dtype(cls, ntrack):
        return np.dtype(cls._dtypes[ntrack])

    @property
    def stream_dtype(self):
        return self.__class__._stream_dtype(self.ntrack)

    @classmethod
    def fromfile(cls, fh, ntrack, decade=None, verify=True):
        """Read Mark 4 header from file.

        Parameters
        ----------
        fh : filehandle
            To read header from.
        ntrack : int
            Number of Mark 4 bitstreams.
        decade : int, or None
            Decade the observations were taken (needed to remove ambiguity in
            the Mark 4 time stamp).
        verify : bool
            Whether to do basic verification of integrity.  Default: `True`.
        """
        dtype = cls._stream_dtype(ntrack)
        size = ntrack * 5 * 32 // 8
        try:
            stream = np.fromstring(fh.read(size), dtype=dtype)
            assert len(stream) * dtype.itemsize == size
        except (ValueError, AssertionError):
            raise EOFError("Could not read full Mark 4 Header.")

        words = stream2words(stream,
                             track=np.arange(ntrack, dtype=stream.dtype))
        self = cls(words, decade=decade, verify=verify)
        self.mutable = False
        return self

    def tofile(self, fh):
        stream = words2stream(self.words)
        fh.write(stream.tostring())

    @classmethod
    def fromvalues(cls, ntrack, decade=None, **kwargs):
        """Initialise a header from parsed values.

        Here, the parsed values must be given as keyword arguments, i.e., for
        any ``header = cls(<words>)``, ``cls.fromvalues(**header) == header``.

        However, unlike for the ``fromkeys`` class method, data can also be set
        using arguments named after header methods such as ``time``.

        Parameters
        ----------
        ntrack : int
            Number of Mark 4 bitstreams.
        decade : int, or None
            Decade the observations were taken (needed to remove ambiguity in
            the Mark 4 time stamp).
        **kwargs :
            Values used to initialize header keys or methods.

        --- Header keywords : (minimum for a complete header)

        time : `~astropy.time.Time` instance
            Sets bcd-encoded unit year, day, hour, minute, second.
        bps : int
            Bits per sample.
        fanout : int
            Number of tracks over which a given channel is spread out. Together
            with ``ntrack`` and ``bps``, this defines ``headstack_id``,
            ``track_id``, ``fan_out``, ``magnitude_bit``, and ``converter_id``.
        """
        # Need to pass on ntrack also as keyword, since the setter is useful.
        kwargs['ntrack'] = ntrack
        return super(Mark4Header, cls).fromvalues(ntrack, decade, **kwargs)

    def update(self, *args, **kwargs):
        """Update the header by setting keywords or properties.

        Here, any keywords matching header keys are applied first, and any
        remaining ones are used to set header properties, in the order set
        by the class (in ``_properties``).

        Parameters
        ----------
        crc : int or `None`, optional
            If `None` (default), recalculate the CRC after updating.
        verify : bool, optional
            If `True` (default), verify integrity after updating.
        **kwargs
            Arguments used to set keywords and properties.
        """
        calculate_crc = kwargs.get('crc', None) is None
        if calculate_crc:
            kwargs.pop('crc', None)
            verify = kwargs.pop('verify', True)
            kwargs['verify'] = False

        super(Mark4Header, self).update(**kwargs)
        if calculate_crc:
            stream = words2stream(self.words)
            stream[-12:] = crc12(stream[:-12])
            self.words = stream2words(stream)
            if verify:
                self.verify()

    @property
    def ntrack(self):
        return self.words.shape[1]

    @ntrack.setter
    def ntrack(self, ntrack):
        assert ntrack == self.words.shape[1]
        if ntrack == 64:
            self['headstack_id'] = np.repeat(np.arange(2), 32)
            self.track_id = np.repeat(np.arange(2, 34)[np.newaxis, :],
                                      2, axis=0).ravel()
        else:
            raise ValueError("Only can set ntrack=64 so far.")

    @property
    def size(self):
        return self.ntrack * 160 // 8

    @property
    def framesize(self):
        return self.ntrack * PAYLOADSIZE // 8

    @framesize.setter
    def framesize(self, framesize):
        assert framesize * 8 % PAYLOADSIZE == 0
        self.ntrack = framesize * 8 // PAYLOADSIZE

    @property
    def payloadsize(self):
        """Payloadsize; missing pieces are the header bytes."""
        return self.framesize - self.size

    @payloadsize.setter
    def payloadsize(self, payloadsize):
        self.framesize = payloadsize + self.size

    @property
    def fanout(self):
        return np.max(self['fan_out']) + 1

    @fanout.setter
    def fanout(self, fanout):
        assert fanout in (1, 2, 4)
        # fanout = 4: (0,0,1,1,2,2,3,3) * 8
        # fanout = 2: (0,0,1,1) * 16
        # fanout = 1: (0,0) * 32
        self['fan_out'] = np.repeat(
            np.repeat(np.arange(fanout), 2)[np.newaxis, :],
            self.ntrack // 2 // fanout, axis=0).ravel()

    @property
    def samples_per_frame(self):
        """Number of samples per channel encoded in frame."""
        # Header overwrites part of payload, so we need framesize.
        # framesize * 8 // bps // nchan, but use ntrack and fanout, as these
        # are more basic; ntrack / fanout by definition equals bps * nchan.
        return self.framesize * 8 // (self.ntrack // self.fanout)

    @samples_per_frame.setter
    def samples_per_frame(self, samples_per_frame):
        self.fanout = samples_per_frame * self.ntrack // 8 // self.framesize

    @property
    def bps(self):
        return 1 if not np.any(self['magnitude_bit']) else 2

    @bps.setter
    def bps(self, bps):
        if bps == 1:
            self['magnitude_bit'] = False
        elif bps == 2:
            self['magnitude_bit'] = np.repeat(
                np.repeat(np.array([False, True]),
                          self.fanout * 2)[np.newaxis, :],
                self.ntrack // 4 // self.fanout, axis=0).ravel()
        else:
            raise ValueError("Mark 4 data only supports 1 or 2 bits/sample")

        nchan = self.ntrack // self.fanout // bps
        self['converter_id'] = np.repeat(
            np.arange(nchan).reshape(-1, 2, 2).transpose(0, 2, 1),
            self.ntrack // nchan, axis=1).ravel()

    @property
    def nchan(self):
        return self.ntrack // self.fanout // self.bps

    @nchan.setter
    def nchan(self, nchan):
        self.bps = self.ntrack // self.fanout // nchan

    def get_time(self):
        """
        Convert BCD time code to Time object for all tracks.

        If all tracks have the same fractional seconds, only a single Time
        instance is returned.

        Uses bcd-encoded 'unit_year', 'day', 'hour', 'minute', 'second' and
        'frac_sec', plus ``decade`` from the initialisation to calculate the
        time.  See http://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf
        """
        if len(set(self['bcd_fraction'])) == 1:
            return self[0].time
        else:
            return Time([h.time for h in self], precision=5)

    def set_time(self, time):
        if time.isscalar:
            super(Mark4Header, self).set_time(time)
        else:
            decades = set()
            for h, t in zip(self, time):
                h.set_time(t)
                decades.add(h.decade)
            if len(decades) > 1:
                raise ValueError("MarkHeader cannot have tracks that differ "
                                 "in the decade of the time they were taken.")
            self.decade = decades.pop()

    time = property(get_time, set_time)

    def __len__(self):
        return self.ntrack

    def __getitem__(self, item):
        if isinstance(item, six.string_types):
            return super(Mark4Header, self).__getitem__(item)

        try:
            new_words = self.words[:, item]
        except IndexError:
            raise IndexError("Index {item} is out of bounds.")

        if not(1 <= new_words.ndim <= 2 and new_words.shape[0] == 5):
            raise ValueError("Cannot extract {0} from {1} instance."
                             .format(item, type(self)))

        if new_words.ndim == 1:
            return self._track_header(new_words, self.decade,
                                      verify=False)
        else:
            return self.__class__(new_words, self.decade, verify=False)

    def __eq__(self, other):
        return (type(self) is type(other) and
                np.all(self.words == other.words))

    def __repr__(self):
        name = self.__class__.__name__
        outs = []
        for k in self.keys():
            v = self[k]
            if len(v) == 1:
                outs.append('{0}: {1}'.format(
                    k, hex(v[0]) if self._repr_as_hex(k) else v[0]))
            elif np.all(v == v[0]):
                outs.append('{0}: [{1}]*{2}'.format(
                    k, hex(v[0]) if self._repr_as_hex(k) else v[0], v.size))
            else:
                if len(v) > 4:
                    v = (v[0], '...', v[-1])
                outs.append('{0}: [{1}]'.format(k, ', '.join(
                    (hex(_v) if _v != '...' and self._repr_as_hex(k)
                     else str(_v)) for _v in v)))

        return "<{0} {1}>".format(name,
                                  (",\n  " + len(name) * " ").join(outs))
