"""
Definitions for VLBI Mark 4 Headers.

Implements a Mark4Header class used to store header words, and decode/encode
the information therein.

For the specification of tape Mark 4 format, see
http://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf

A little bit on the disk representation is at
http://adsabs.harvard.edu/abs/2003ASPC..306..123W
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from astropy.time import Time
from astropy.extern import six

from ..vlbi_base.header import HeaderParser, VLBIHeaderBase
from ..vlbi_base.utils import bcd_decode, bcd_encode, CRC

__all__ = ['CRC12', 'crc12', 'stream2words', 'words2stream',
           'Mark4TrackHeader', 'Mark4Header']


MARK4_DTYPES = {8: '<u1',  # this needs to start with '<' for words2stream.
                16: '<u2',
                32: '<u4',
                64: '<u8'}
"""Integer dtype used to encode a given number of tracks."""

PAYLOAD_NBITS = 20000
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
    stream : `~numpy.array` of int
        For each int, every bit corresponds to a particular track.
    track : int, array, or None, optional
        The track to extract.  If `None` (default), extract all tracks that
        the type of int in the stream can hold.
    """
    if track is None:
        track = np.arange(stream.dtype.itemsize * 8, dtype=stream.dtype)

    track_sel = ((stream.reshape(-1, 32, 1) >> track) & 1).astype(np.uint32)
    track_sel <<= np.arange(31, -1, -1, dtype=np.uint32).reshape(-1, 1)
    words = np.bitwise_or.reduce(track_sel, axis=1)
    return words.squeeze()


def words2stream(words):
    """Convert a set of uint32 header words to a stream of integers.

    Parameters
    ----------
    words : `~numpy.array` of uint32

    Returns
    -------
    stream : `~numpy.array` of int
        For each int, every bit corresponds to a particular track.
    """
    ntrack = words.shape[1]
    dtype = MARK4_DTYPES[ntrack]
    nbits = words.dtype.itemsize * 8
    bit = np.arange(nbits - 1, -1, -1, dtype=words.dtype).reshape(-1, 1)

    bit_sel = ((words[:, np.newaxis, :] >> bit) & 1).astype(dtype[1:])
    bit_sel <<= np.arange(ntrack, dtype=dtype[1:])
    words = np.empty(bit_sel.shape[:2], dtype)
    words = np.bitwise_or.reduce(bit_sel, axis=2, out=words)
    return words.ravel()


class Mark4TrackHeader(VLBIHeaderBase):
    """Decoder/encoder of a Mark 4 Track Header.

    See http://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf

    Parameters
    ----------
    words : tuple of int, or None
        Five 32-bit unsigned int header words.  If `None`, set to a list of
        zeros for later initialisation.
    decade : int or None
        Decade in which the observations were taken (needed to remove ambiguity
        in the Mark 4 time stamp).  Can instead pass an approximate
        ``ref_time``.
    ref_time : `~astropy.time.Time` or None
        Reference time within 4 years of the observation time, used to infer
        the full Mark 4 timestamp.  Used only if ``decade`` is not given.
    verify : bool, optional
        Whether to do basic verification of integrity.  Default: `True`.

    Returns
    -------
    header : `~baseband.mark4.header.Mark4TrackHeader`
    """

    _header_parser = HeaderParser(
        (('bcd_headstack1', (0, 0, 16, 0x3344)),
         ('bcd_headstack2', (0, 16, 16, 0x1122)),
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
    _sync_pattern = _header_parser.defaults['sync_pattern']

    _properties = ('decade', 'track_id', 'fraction', 'time')
    """Properties accessible/usable in initialisation."""

    decade = None

    def __init__(self, words, decade=None, ref_time=None, verify=True):
        if words is None:
            self.words = [0, 0, 0, 0, 0]
        else:
            self.words = words
        if decade is not None:
            self.decade = decade
        elif ref_time is not None:
            self.infer_decade(ref_time)
        if verify:
            self.verify()

    def verify(self):
        """Verify header integrity."""
        assert len(self.words) == 5
        assert np.all(self['sync_pattern'] == self._sync_pattern)
        assert np.all((self['bcd_fraction'] & 0xf) % 5 != 4)
        if self.decade is not None:
            assert (1950 < self.decade < 3000)
            assert self.decade % 10 == 0, "decade must end in zero"

    def infer_decade(self, ref_time):
        """Uses a reference time to set a header's ``decade``.

        Parameters
        ----------
        ref_time : `~astropy.time.Time`
            Reference time within 5 years of the observation time.
        """
        self.decade = np.round(ref_time.decimalyear - self['bcd_unit_year'],
                               decimals=-1).astype(int)

    @property
    def track_id(self):
        return bcd_decode(self['bcd_track_id'])

    @track_id.setter
    def track_id(self, track_id):
        self['bcd_track_id'] = bcd_encode(track_id)

    @property
    def fraction(self):
        """Fractional seconds (decoded from 'bcd_fraction')."""
        ms = bcd_decode(self['bcd_fraction'])
        # The last digit encodes a fraction -- see table 2 in
        # http://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf
        # 0: 0.00      5: 5.00
        # 1: 1.25      6: 6.25
        # 2: 2.50      7: 7.50
        # 3: 3.75      8: 8.75
        # 4: invalid   9: invalid
        last_digit = ms % 5
        return (ms + last_digit * 0.25) / 1000.

    @fraction.setter
    def fraction(self, fraction):
        ms = fraction * 1000.
        if np.any(np.abs((ms / 1.25) - np.round(ms / 1.25)) > 1e-6):
            raise ValueError("{0} ms is not a multiple of 1.25 ms"
                             .format(ms))
        self['bcd_fraction'] = bcd_encode(np.floor(ms + 1e-6)
                                          .astype(np.int32))

    def get_time(self):
        """Convert BCD time code to Time object.

        Calculate time using bcd-encoded 'bcd_unit_year', 'bcd_day',
        'bcd_hour', 'bcd_minute', 'bcd_second' header items, as well as
        the ``fraction`` property (inferred from 'bcd_fraction') and
        ``decade`` from the initialisation.  See
        See http://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf
        """
        return Time('{decade:03d}{uy:1x}:{d:03x}:{h:02x}:{m:02x}:{s:08.5f}'
                    .format(decade=self.decade//10, uy=self['bcd_unit_year'],
                            d=self['bcd_day'], h=self['bcd_hour'],
                            m=self['bcd_minute'],
                            s=bcd_decode(self['bcd_second']) + self.fraction),
                    format='yday', scale='utc', precision=5)

    def set_time(self, time):
        old_precision = time.precision
        try:
            time.precision = 5
            yday = time.yday.split(':')
        finally:
            time.precision = old_precision
        # Set fraction first since that checks precision.
        self.fraction = float(yday[4]) % 1
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
    words : `~numpy.ndarray` of int, or None
        Shape should be (5, number-of-tracks), and dtype np.uint32.  If `None`,
        ``ntrack`` should be given and words will be initialized to 0.
    ntrack : None or int
        Number of Mark 4 bitstreams, to help initialize ``words`` if needed.
    decade : int or None
        Decade in which the observations were taken (needed to remove ambiguity
        in the Mark 4 time stamp).  Can instead pass an approximate
        ``ref_time``.
    ref_time : `~astropy.time.Time` or None
        Reference time within 4 years of the observation time, used to infer
        the full Mark 4 timestamp.  Used only if ``decade`` is not given.
    verify : bool, optional
        Whether to do basic verification of integrity.  Default: `True`.

    Returns
    -------
    header : `~baseband.mark4.Mark4Header`
    """

    _track_header = Mark4TrackHeader
    _properties = (Mark4TrackHeader._properties +
                   ('fanout', 'samples_per_frame', 'bps', 'nchan', 'nsb',
                    'converters'))
    _dtypes = MARK4_DTYPES

    # keyed with bps, fanout; Tables 10-14 in reference documentation:
    # http://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf
    # rows are channels with Sign, Mag for each for bps=2, columns fanout.
    # So for bps=2, fanout=4 (abbreviating channel a Sign, Mag as aS, aM):
    # Channel a has samples (aS, aM) in tracks (2, 10), (4,12), etc.
    #         b             (bS, bM) in        (3, 11), etc.
    # We subtract two and reshape as (fanout, nchan, bps) since that is how
    # it is used internally.
    _track_assignments = {
        (2, 4): np.array(  # rows=aS, aM, bS, bM, cS, cM, dS, dM; cols=fanout.
            [[2, 10, 3, 11, 18, 26, 19, 27],
             [4, 12, 5, 13, 20, 28, 21, 29],
             [6, 14, 7, 15, 22, 30, 23, 31],
             [8, 16, 9, 17, 24, 32, 25, 33]]).reshape(4, 4, 2) - 2,
        (1, 4): np.array(  # rows=aS, bS, ..., hS; cols=fanout.
            [[2, 3, 10, 11, 18, 19, 26, 27],
             [4, 5, 12, 13, 20, 21, 28, 29],
             [6, 7, 14, 15, 22, 23, 30, 31],
             [8, 9, 16, 17, 24, 25, 32, 33]]).reshape(4, 8, 1) - 2,
        (2, 2): (np.array(  # rows=aS, aM, bS, bM, ..., hS, hM; cols=fanout.
            [[2, 6, 3, 7, 10, 14, 11, 15, 18, 22, 19, 23, 26, 30, 27, 31],
             [4, 8, 5, 9, 12, 16, 13, 17, 20, 24, 21, 25, 28, 32, 29, 33]])
                 .reshape(2, 8, 2) - 2),
        (1, 2): (np.array(  # rows=aS, bS, ..., pS; cols=fanout.
            [[2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27, 30, 31],
             [4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33]])
                 .reshape(2, 16, 1) - 2),
        (2, 1): (np.array(  # rows=aS, aM, bS, bM, ..., pS, pM; no fanout.
            [[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
              3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33]])
                 .reshape(1, 16, 2) - 2)}

    def __init__(self, words, ntrack=None, decade=None, ref_time=None,
                 verify=True):
        if words is None:
            words = np.zeros((5, ntrack), dtype=np.uint32)
        super(Mark4Header, self).__init__(words, decade=decade,
                                          ref_time=ref_time, verify=verify)

    def verify(self):
        super(Mark4Header, self).verify()
        assert set(self['fan_out']) == set(np.arange(self.fanout))
        assert (len(set((c, l) for (c, l) in zip(self['converter_id'],
                                                 self['lsb_output']))) ==
                self.nchan)

    def infer_decade(self, ref_time):
        super(Mark4Header, self).infer_decade(ref_time)
        if getattr(self.decade, 'size', 1) > 1:
            assert np.all(self.decade == self.decade[0])
            self.decade = self.decade[0]

    @classmethod
    def _stream_dtype(cls, ntrack):
        return np.dtype(cls._dtypes[ntrack])

    @property
    def stream_dtype(self):
        """Stream dtype required to hold this header's number of tracks."""
        return self._stream_dtype(self.ntrack)

    @classmethod
    def _track_assignment(cls, ntrack, bps, fanout):
        try:
            ta = cls._track_assignments[(bps, fanout)]
        except KeyError:
            raise ValueError("Mark 4 reader does not support bps={0}, "
                             "fanout={1}; supported are {2}".format(
                                 bps, fanout, cls._track_assignments.keys()))

        if ntrack == 64:
            # double up the number of tracks and channels.
            return np.concatenate((ta, ta + 32), axis=1)
        elif ntrack == 32:
            return ta
        elif ntrack == 16:
            return ta[:, ::2, :] // 2
        else:
            raise ValueError("have Mark 4 track assignments only for "
                             "ntrack=32 or 64, not {0}".format(ntrack))

    @property
    def track_assignment(self):
        """Assignments of tracks to channels and fanout items.

        The assignments are inferred from tables 10-14 in
        http://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf
        except that 2 has been subtracted so that tracks start at 0,
        and that for 64 tracks the arrays are suitably enlarged by adding
        another set of channels.

        The returned array has shape ``(fanout, nchan, bps)``.
        """
        return self._track_assignment(self.ntrack, self.bps, self.fanout)

    @classmethod
    def fromfile(cls, fh, ntrack, decade=None, ref_time=None, verify=True):
        """Read Mark 4 header from file.

        Parameters
        ----------
        fh : filehandle
            To read header from.
        ntrack : int
            Number of Mark 4 bitstreams.
        decade : int or None
            Decade in which the observations were taken.  Can instead pass an
            approximate ``ref_time``.
        ref_time : `~astropy.time.Time` or None
            Reference time within 4 years of the observation time.  Used only
            if ``decade`` is not given.
        verify : bool, optional
            Whether to do basic verification of integrity.  Default: `True`.
        """
        dtype = cls._stream_dtype(ntrack)
        header_nbytes = ntrack * 160 // 8
        try:
            stream = np.frombuffer(fh.read(header_nbytes), dtype=dtype)
            assert len(stream) * dtype.itemsize == header_nbytes
        except (ValueError, AssertionError):
            raise EOFError("could not read full Mark 4 Header.")

        words = stream2words(stream)
        self = cls(words, decade=decade, ref_time=ref_time, verify=verify)
        self.mutable = False
        return self

    def tofile(self, fh):
        stream = words2stream(self.words)
        fh.write(stream.tostring())

    @classmethod
    def fromvalues(cls, ntrack, decade=None, ref_time=None, **kwargs):
        """Initialise a header from parsed values.

        Here, the parsed values must be given as keyword arguments, i.e., for
        any ``header = cls(<words>)``, ``cls.fromvalues(**header) == header``.

        However, unlike for the `fromkeys` class method, data can also be set
        using arguments named after header methods, such as ``time``.

        Parameters
        ----------
        ntrack : int
            Number of Mark 4 bitstreams.
        decade : int or None, optional
            Decade in which the observations were taken.  Can instead pass an
            approximate ``ref_time``.  Not needed if ``time`` is given.
        ref_time : `~astropy.time.Time` or None, optional
            Reference time within 4 years of the observation time.  Used only
            if ``decade`` is not given, and not needed if ``time`` is given.
        **kwargs :
            Values used to initialize header keys or methods.

        --- Header keywords : (minimum for a complete header)

        time : `~astropy.time.Time` instance
            Time of the first sample.
        bps : int
            Bits per elementary sample.
        fanout : int
            Number of tracks over which a given channel is spread out.
        """
        # set defaults based on ntrack for cases where it is known.
        if ntrack == 64:
            kwargs.setdefault('headstack_id', np.repeat(np.arange(2), 32))
            kwargs.setdefault('track_id', np.tile(np.arange(2, 34), 2))
        elif ntrack == 32:
            kwargs.setdefault('headstack_id', np.zeros(32, dtype=int))
            kwargs.setdefault('track_id', np.arange(2, 34))
        elif ntrack == 16:
            kwargs.setdefault('headstack_id', np.zeros(16, dtype=int))
            kwargs.setdefault('track_id', np.arange(2, 34, 2))
        # set number of sidebands to default if no information is given,
        # so that the header will be valid.
        if not any(key in kwargs for key in ('lsb_output', 'converter_id',
                                             'converter')):
            kwargs.setdefault('nsb', 1)
        return super(Mark4Header, cls).fromvalues(ntrack, decade, ref_time,
                                                  **kwargs)

    def update(self, crc=None, verify=True, **kwargs):
        """Update the header by setting keywords or properties.

        Here, any keywords matching header keys are applied first, and any
        remaining ones are used to set header properties, in the order set
        by the class (in ``_properties``).

        Parameters
        ----------
        crc : int or None, optional
            If `None` (default), recalculate the CRC after updating.
        verify : bool, optional
            If `True` (default), verify integrity after updating.
        **kwargs
            Arguments used to set keywords and properties.
        """
        if crc is None:
            super(Mark4Header, self).update(verify=False, **kwargs)
            stream = words2stream(self.words)
            stream[-12:] = crc12(stream[:-12])
            self.words = stream2words(stream)
            if verify:
                self.verify()
        else:
            super(Mark4Header, self).update(verify=verify, crc=crc, **kwargs)

    @property
    def ntrack(self):
        """Number of Mark 4 bitstreams."""
        return self.words.shape[1]

    @property
    def nbytes(self):
        """Size of the header in bytes."""
        return self.ntrack * 160 // 8

    @property
    def frame_nbytes(self):
        """Size of the frame in bytes."""
        return self.ntrack * PAYLOAD_NBITS // 8

    @property
    def payload_nbytes(self):
        """Size of the payload in bytes.

        Note that the payloads miss pieces overwritten by the header.
        """
        return self.frame_nbytes - self.nbytes

    @property
    def fanout(self):
        """Number of samples stored in one payload item of size ntrack.

        If set, will update 'fan_out' for each track.
        """
        return np.max(self['fan_out']) + 1

    @fanout.setter
    def fanout(self, fanout):
        if fanout not in (1, 2, 4):
            raise ValueError("Mark 4 data only supports fanout=1, 2, or 4, "
                             "not {0}.".format(fanout))
        # In principle, one would like to go through track_assignments, but
        # we may not have bps set here yet, so just infer from tables:
        # fanout = 4: (0,1,2,3) * ntrack / 4              if ntrack = 16
        #             (0,0,1,1,2,2,3,3) * ntrack / 2 / 4  otherwise
        # fanout = 2: (0,0,1,1) * ntrack / 2 / 2
        # fanout = 1: (0,0) * ntrack / 2
        if self.ntrack == 16:
            self['fan_out'] = np.tile(np.arange(fanout), self.ntrack // fanout)
        else:
            self['fan_out'] = np.tile(np.repeat(np.arange(fanout), 2),
                                      self.ntrack // 2 // fanout)

    @property
    def samples_per_frame(self):
        """Number of complete samples in the frame.

        If set, this uses the number of tracks to infer and set `fanout`.
        """
        # Header overwrites part of payload, so we need
        # frame_nbytes * 8 // bps // nchan, but use ntrack and fanout, as these
        # are more basic; ntrack / fanout by definition equals bps * nchan.
        return self.frame_nbytes * 8 // (self.ntrack // self.fanout)

    @samples_per_frame.setter
    def samples_per_frame(self, samples_per_frame):
        self.fanout = samples_per_frame * self.ntrack // 8 // self.frame_nbytes

    @property
    def bps(self):
        """Bits per elementary sample (either 1 or 2).

        If set, combined with `fanout` and `ntrack` to update 'magnitude_bit'
        for all tracks.
        """
        return 2 if self['magnitude_bit'].any() else 1

    @bps.setter
    def bps(self, bps):
        if bps == 1:
            self['magnitude_bit'] = False
        elif bps == 2:
            # Note: cannot assign to slice of header property, so go via array.
            ta = self._track_assignment(self.ntrack, bps, self.fanout)
            magnitude_bit = np.empty(self.ntrack, dtype=bool)
            magnitude_bit[ta] = [False, True]
            self['magnitude_bit'] = magnitude_bit
        else:
            raise ValueError("Mark 4 data can only have bps=1 or 2, "
                             "not {0}".format(bps))

    @property
    def nchan(self):
        """Number of channels (``ntrack * fanout``) in the frame.

        If set, it is combined with `ntrack` and `fanout` to infer `bps`.
        """
        return self.ntrack // (self.fanout * self.bps)

    @nchan.setter
    def nchan(self, nchan):
        self.bps = self.ntrack // (self.fanout * nchan)

    @property
    def nsb(self):
        """Number of side bands used.

        If set, assumes all converters are upper sideband for 1, and that
        converter IDs alternate between upper and lower sideband for 2.
        """
        sb = self['lsb_output']
        return 1 if (sb == sb[0]).all() else 2

    @nsb.setter
    def nsb(self, nsb):
        if nsb == 1:
            self['lsb_output'] = True

        elif nsb == 2:
            ta = self.track_assignment
            ta_ch = ta[0, :, 0]
            sb = np.tile([False, True], len(ta_ch) // 2)
            lsb_output = np.empty(self.ntrack, bool)
            lsb_output[ta] = sb[:, np.newaxis]
            self['lsb_output'] = np.tile([False, True], 16)

        else:
            raise ValueError("number of sidebands can only be 1 or 2.")

        # Set default converters; can be overridden if needed.
        nconverter = self.ntrack // (self.fanout * self.bps * self.nsb)
        converters = np.arange(nconverter)
        if nconverter > 2:
            converters = (converters.reshape(-1, 2, 2)
                          .transpose(0, 2, 1).ravel())
        self.converters = converters

    @property
    def converters(self):
        """Converted ID and sideband used for each channel.

        Returns a structured array with numerical 'converter' and boolean
        'lsb' entries (where `True` means lower sideband).

        Can be set with a similar structured array or a `dict`; if just an
        an array is passed in, it will be assumed that the sideband has been
        set beforehand (e.g., by setting `nsb`) and that the array holds
        the converter IDs.
        """
        ta_ch = self.track_assignment[0, :, 0]
        converters = np.empty(len(ta_ch), [("converter", int), ("lsb", bool)])
        converters['converter'] = self['converter_id'][ta_ch]
        converters['lsb'] = self['lsb_output'][ta_ch]
        return converters

    @converters.setter
    def converters(self, converters):
        # Set converters, duplicating over fanout, lsb, magnitude bit.
        ta = self.track_assignment
        ta_ch = ta[0, :, 0]
        nchan = len(ta_ch)
        msg = ('Mark 4 file with bps={0}, fanout={1} '
               'needs to define {2} converters')
        try:
            converter = converters['converter']
        except(KeyError, ValueError, IndexError):
            converter = np.array(converters)
            sb = self['lsb_output'][ta_ch]
            if self.nsb == 2 and len(converter) == len(ta_ch) // 2:
                c = np.empty(len(ta_ch), dtype=int)
                c[sb] = c[~sb] = converter
                converter = c
            if len(converter) != nchan:
                raise ValueError(msg.format(self.bps, self.fanout, nchan))

        else:
            sb = np.array(converters['lsb'])
            if len(converter) != nchan:
                raise ValueError(msg.format(self.bps, self.fanout, nchan))
            lsb_output = np.empty(self.ntrack, bool)
            lsb_output[ta] = sb[:, np.newaxis]
            self['lsb_output'] = lsb_output

        # Note: cannot assign to slice of header property, so go via array.
        converter_id = np.empty(self.ntrack, dtype=int)
        converter_id[ta] = converter[:, np.newaxis]
        self['converter_id'] = converter_id

    def get_time(self):
        """Convert BCD time code to Time object for all tracks.

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
                raise ValueError("Mark4Header cannot have tracks that differ "
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
            raise IndexError("index {item} is out of bounds.")

        if not(1 <= new_words.ndim <= 2 and new_words.shape[0] == 5):
            raise ValueError("cannot extract {0} from {1} instance."
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
