# Licensed under the GPLv3 - see LICENSE
"""
Definitions for GSB Headers, using the timestamp files.

Somewhat out of data description for phased data:
http://gmrt.ncra.tifr.res.in/gmrt_hpage/sub_system/gmrt_gsb/GSB_beam_timestamp_note_v1.pdf
and for rawdump data
http://gmrt.ncra.tifr.res.in/gmrt_hpage/sub_system/gmrt_gsb/GSB_rawdump_data_format_v2.pdf
"""
import os

import numpy as np
from astropy import units as u
from astropy.time import Time, TimeString
from astropy.time.formats import erfa  # astropy-version independent.

from ..base.header import HeaderParserBase, ParsedHeaderBase, ParserDict


__all__ = ['TimeGSB', 'GSBHeader', 'GSBRawdumpHeader', 'GSBPhasedHeader']


class TimeGSB(TimeString):
    """GSB header date-time format ``YYYY MM DD HH MM SS 0.SSSSSSSSS``.

    For example, ``2000 01 01 00 00 00 0.000000000`` is midnight on
    January 1, 2000.
    """
    # Implicitly uses the metaclass astropy.time.formats.TimeFormatMeta to
    # register with astropy.Time.

    name = 'gsb'

    def set_jds(self, val1, val2):
        """Parse the time strings contained in val1 and set jd1, jd2"""
        iterator = np.nditer([val1, None, None, None, None, None, None],
                             op_dtypes=([val1.dtype] + 5 * [np.intc]
                                        + [np.double]))
        try:
            for val, iy, im, id, ihr, imin, dsec in iterator:
                timestr = val.item()
                components = timestr.split()
                iy[...], im[...], id[...], ihr[...], imin[...], sec = (
                    int(component) for component in components[:-1])
                dsec[...] = sec + float(components[-1])
        except Exception:
            raise ValueError('Time {0} does not match {1} format'
                             .format(timestr, self.name))

        self.jd1, self.jd2 = erfa.dtf2d(
            self.scale.upper().encode('utf8'), *iterator.operands[1:])

    def to_value(self, parent=None):
        scale = self.scale.upper().encode('ascii'),
        iys, ims, ids, ihmsfs = erfa.d2dtf(scale, self.precision,
                                           self.jd1, self.jd2)
        ihrs = ihmsfs['h']
        imins = ihmsfs['m']
        isecs = ihmsfs['s']
        ifracs = ihmsfs['f']

        fmt = ('{0:04d} {1:02d} {2:02d} {3:02d} {4:02d} {5:02d} '
               '0.{6:0' + str(self.precision) + 'd}')
        outs = []
        for iy, im, id, ihr, imin, isec, ifracsec in np.nditer(
                [iys, ims, ids, ihrs, imins, isecs, ifracs]):
            outs.append(fmt.format(int(iy), int(im), int(id), int(ihr),
                                   int(imin), int(isec), int(ifracsec)))

        return np.array(outs).reshape(self.jd1.shape)

    value = property(to_value)


def make_parser(index, length, forward=None, backward=None, default=None):
    if length > 1:
        index = slice(index, index + length)

    def parser(items):
        return forward(items[index])

    return parser


def make_setter(index, length, forward, backward, default=None):
    def setter(items, value):
        value = backward(value)
        if length == 1:
            items[index] = value
        else:
            for i, val in enumerate(value):
                items[index + i] = val
        return items

    return setter


def get_default(index, length, forward, backward, default=None):
    return default


class GSBHeaderParser(HeaderParserBase):
    """Parser & setter for GSB timestamp keywords.

    A dictionary of header keywords, with values that describe how they are
    encoded in the GSB header.  Initialisation is as a normal dict,
    with (ordered) key, value pairs, with each value a tuple containing:

    index : int
        Index into the header words for this key.
    length : int
        Number of words included in this key.
    forward : callable
        Function to decode the data.
    backward : callable
        Function to encode the value.
    default : object or None
        Possible default value.

    Notes
    -----
    For GSB, this parsing technique is a bit of overkill, but it allows re-use
    of handy methods from the VLBI header parser.
    """

    parsers = ParserDict(make_parser)
    setters = ParserDict(make_setter)
    defaults = ParserDict(get_default)


class GSBHeader(ParsedHeaderBase):
    """GSB Header, based on a line from a timestamp file.

    Parameters
    ----------
    words : list of str, or None
        If `None`, set to a list of empty strings for later initialisation.
    mode : str or None, optional
        Mode in which data was taken: 'phased' or 'rawdump'. If `None`, it
        is determined from the words.
    nbytes : int or None, optional
        Number of characters in the header, including trailing blank spaces and
        carriage returns.  If `None`, is determined from the words assuming
        one trailing blank space and one CR.
    verify : bool, optional
        Whether to do basic verification of integrity.  Default: `True`.

    Returns
    -------
    header : `GSBHeader` subclass
        As appropriate for the mode.
    """
    _mode = None
    _gsb_header_classes = {}

    def __new__(cls, words=None, mode=None, nbytes=None, utc_offset=5.5*u.hr,
                verify=True):

        if cls is GSBHeader:
            if mode is None:
                if words is None:
                    raise TypeError("cannot construct an empty GSB header "
                                    "without knowing the mode.")

                mode = 'rawdump' if len(words) == 7 else 'phased'

            cls = cls._gsb_header_classes.get(mode)

        # We intialise GSBHeader subclasses, so their __init__ will be called.
        return super().__new__(cls)

    def __init__(self, words, mode=None, nbytes=None, utc_offset=5.5*u.hr,
                 verify=True):
        if words is None:
            words = [''] * self._number_of_words
        if mode is not None:
            self._mode = mode
        self._nbytes = nbytes
        self.utc_offset = utc_offset
        super().__init__(words, verify=verify)

    def verify(self):
        assert self.mode == self.__class__._mode
        assert len(self.words) == self._number_of_words

    @property
    def mode(self):
        """Mode in which data was taken: 'phased' or 'rawdump'."""
        return self._mode

    @property
    def nbytes(self):
        """Size of the header in characters.

        Assumes the string terminates in one blank space and one carriage
        return.
        """
        if self._nbytes is None:
            self._nbytes = len(' '.join(self.words) + os.linesep)
        return self._nbytes

    @classmethod
    def fromfile(cls, fh, *args, **kwargs):
        """Read GSB Header from a line from a timestamp file.

        Arguments are the same as for class initialisation.  The header
        constructed will be immutable.
        """
        start_pos = fh.tell()
        s = fh.readline()
        if s == '':
            raise EOFError
        nbytes = fh.tell() - start_pos
        return cls(tuple(s.split()), mode=None, nbytes=nbytes, *args, **kwargs)

    def tofile(self, fh):
        """Write GSB header as a line to the filehandle."""
        return fh.write(' '.join(self.words) + '\n')

    @classmethod
    def fromvalues(cls, mode=None, nbytes=None, *args, **kwargs):
        if mode is None and cls._mode is None:
            if set(kwargs.keys()) & {'pc', 'pc_time',
                                     'seq_nr', 'mem_block'}:
                mode = 'phased'
            else:
                raise TypeError("cannot construct a GSB header from "
                                "values without knowing the mode.")
        return super().fromvalues(mode, nbytes, *args, **kwargs)

    @classmethod
    def fromkeys(cls, mode=None, nbytes=None, *args, **kwargs):
        if mode is None and cls._mode is None:
            if set(kwargs.keys()) & {'pc', 'seq_nr', 'mem_block'}:
                mode = 'phased'
            else:
                mode = 'rawdump'
        return super().fromkeys(mode, nbytes, *args, **kwargs)

    def seek_offset(self, n, nbytes=None):
        """Offset in bytes needed to move a file pointer to another header.

        Some GSB headers have variable size and hence one cannot trivially jump
        to another entry in a timestamp file.  This routine allows one to
        calculate the offset required to move the file pointer ``n`` headers.

        Parameters
        ----------
        n : int
            The number of headers to move to, relative to the present header.
        nbytes : int, optional
            The size in bytes of the present header (if not given, will use
            the header's `nbytes` property).
        """
        if nbytes is None:
            nbytes = self.nbytes
        return n * nbytes


class GSBRawdumpHeader(GSBHeader):
    """GSB rawdump header."""

    _mode = 'rawdump'
    _number_of_words = 7
    _gps_time_precision = 9
    _properties = ('gps_time', 'time')

    _header_parser = GSBHeaderParser(
        (('gps', (0, 7, ' '.join, str.split)),))

    @property
    def gps_time(self):
        return Time(self['gps'], format='gsb',
                    precision=self._gps_time_precision) - self.utc_offset

    @gps_time.setter
    def gps_time(self, time):
        t = time + self.utc_offset
        t.precision = self._gps_time_precision
        self['gps'] = t.gsb

    time = gps_time


class GSBPhasedHeader(GSBRawdumpHeader):
    """GSB phased header."""

    _mode = 'phased'
    _number_of_words = GSBRawdumpHeader._number_of_words + 7 + 2
    _pc_time_precision = 6
    _properties = ('time', 'pc_time') + GSBRawdumpHeader._properties

    _header_parser = GSBHeaderParser(
        (('pc', (0, 7, ' '.join, str.split)),
         ('gps', (7, 7, ' '.join, str.split)),
         ('seq_nr', (14, 1, int, str, 0)),
         ('mem_block', (15, 1, int, str, 0))))

    @property
    def pc_time(self):
        return Time(self['pc'], format='gsb',
                    precision=self._pc_time_precision) - self.utc_offset

    @pc_time.setter
    def pc_time(self, time):
        t = time + self.utc_offset
        t.precision = self._pc_time_precision
        self['pc'] = t.gsb

    @property
    def time(self):
        return self.gps_time

    @time.setter
    def time(self, time):
        self.gps_time = time
        self.pc_time = time

    def seek_offset(self, n, nbytes=None):
        """Offset in bytes needed to move a file pointer to another header.

        GSB headers for phased data differ in size depending on the sequence
        number, making it impossible to trivially jump to another entry in a
        timestamp file.  This routine allows one to calculate the offset
        required to move the file pointer ``n`` headers.

        Parameters
        ----------
        n : int
            The number of headers to move to, relative to the present header.
        nbytes : int, optional
            The size in bytes of the present header (if not given, will use
            the header's `nbytes` property).
        """
        if nbytes is None:
            nbytes = self.nbytes
        # Initial guess assuming all headers have same nbytes.
        guess = n * nbytes
        # Get number of digits of current sequence number.
        seq = self['seq_nr']
        ndseq = len(str(seq))
        # Find the sequence number we're trying to reach.
        seq_sub_targ = seq + n
        # And get number of digits for this value.
        ndtarg = len(str(seq_sub_targ))
        # If numbers not the same, correct appropriately.
        while ndseq != ndtarg:
            if n > 0:
                next_power_of_ten = int('1' + ndseq * '0')
                guess += seq_sub_targ - next_power_of_ten
                ndseq += 1
            else:
                next_power_of_ten = int('1' + (ndseq - 1) * '0')
                guess += next_power_of_ten - seq_sub_targ
                ndseq -= 1

        return guess


GSBHeader._gsb_header_classes.update(rawdump=GSBRawdumpHeader,
                                     phased=GSBPhasedHeader)
