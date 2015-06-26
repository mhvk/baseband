"""
Definitions for VLBI Mark5B Headers.

Implements a Mark5BHeader class used to store header words, and decode/encode
the information therein.

For the specification, see
http://www.haystack.edu/tech/vlbi/mark5/docs/Mark%205B%20users%20manual.pdf
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.time import Time

from ..vlbi_base import (HeaderParser, VLBIHeaderBase, four_word_struct,
                         bcd_decode, bcd_encode)


class Mark5BHeader(VLBIHeaderBase):
    """Decoder/encoder of a Mark5B Frame Header.

    See page 15 of
    http://www.haystack.edu/tech/vlbi/mark5/docs/Mark%205B%20users%20manual.pdf

    Parameters
    ----------
    words : tuple of int, or None
        Eight (or four for legacy VDIF) 32-bit unsigned int header words.
        If ``None``, set to a tuple of zeros for later initialisation.
    ref_mjd : float, or None
        MJD within 500 days of the observation time, used to infer the
        full MJD from the time information in the header (which only has
        last 3 digits of the MJD).
    kday : int, or None
        Explicit thousands of MJD.
    verify : bool
        Whether to do basic verification of integrity.  Default: `True`.

    Returns
    -------
    header : Mark5BHeader instance.
    """

    _header_parser = HeaderParser(
        (('sync_pattern', (0, 0, 32, 0xABADDEED)),
         ('user', (1, 16, 16)),
         ('internal_tvg', (1, 15, 1)),
         ('frame_nr', (1, 0, 15)),
         ('bcd_jday', (2, 20, 12)),
         ('bcd_seconds', (2, 0, 20)),
         ('bcd_fraction', (3, 16, 16)),
         ('crcc', (3, 0, 16))))

    _struct = four_word_struct

    _properties = ('payloadsize', 'framesize', 'kday', 'jday', 'seconds',
                   'ns', 'time')
    """Properties accessible/usable in initialisation."""

    kday = None

    def __init__(self, words, ref_mjd=None, kday=None, verify=True):
        if words is None:
            self.words = (0, 0, 0, 0)
        else:
            self.words = words
        if kday is not None:
            self.kday = kday
        elif ref_mjd is not None:
            ref_kday, ref_jday = divmod(ref_mjd, 1000)
            self.kday = int(ref_kday +
                            round((ref_jday - self.jday)/1000)) * 1000
        if verify:
            self.verify()

    def verify(self):
        """Verify header integrity."""
        assert len(self.words) == 4
        assert (self['sync_pattern'] ==
                self._header_parser.defaults['sync_pattern'])
        assert self.kday is not None and (33000 < self.kday < 400000)

    def __repr__(self):
        name = self.__class__.__name__
        return ("<{0} {1}>".format(name, (",\n  " + len(name) * " ").join(
            ["{0}: {1}".format(k, (hex(self[k])
                                   if k.startswith('bcd') or k.startswith('sy')
                                   else self[k])) for k in self.keys()])))

    @property
    def payloadsize(self):
        """Size of the payload, in bytes."""
        return 10000  # 2500 words

    @payloadsize.setter
    def payloadsize(self, payloadsize):
        if payloadsize != 10000:  # 2500 words
            raise ValueError("Mark5B payload has a fixed size of 10000 bytes "
                             "(2500 words).")

    @property
    def framesize(self):
        """Size of a frame, in bytes."""
        return self.size + self.payloadsize

    @framesize.setter
    def framesize(self, framesize):
        if framesize != self.size + self.payloadsize:
            raise ValueError("Mark5B frame has a fixed size of 10016 bytes "
                             "(4 header words plus 2500 payload words).")

    @property
    def jday(self):
        """Last three digits of MJD (decoded from 'bcd_jday')."""
        return bcd_decode(self['bcd_jday'])

    @jday.setter
    def jday(self, jday):
        self['bcd_jday'] = bcd_encode(jday)

    @property
    def seconds(self):
        """Integer seconds on day (decoded from 'bcd_seconds')."""
        return bcd_decode(self['bcd_seconds'])

    @seconds.setter
    def seconds(self, seconds):
        self['bcd_seconds'] = bcd_encode(seconds)

    @property
    def ns(self):
        """Fractional seconds (in ns; decoded from 'bcd_fraction')."""
        ns = bcd_decode(self['bcd_fraction']) * 100000
        # "unround" the nanoseconds
        return 156250 * ((ns+156249) // 156250)

    @ns.setter
    def ns(self, ns):
        fraction = round(ns / 100000)
        self['bcd_fraction'] = bcd_encode(fraction)

    def get_time(self):
        """
        Convert year, BCD time code to Time object.

        Uses bcd-encoded 'jday', 'seconds', and 'frac_sec', plus ``kday``
        from the initialisation to calculate the time.  See
        http://www.haystack.edu/tech/vlbi/mark5/docs/Mark%205B%20users%20manual.pdf
        """
        return Time(self.kday + self.jday,
                    (self.seconds + 1.e-9 * self.ns) / 86400,
                    format='mjd', scale='utc', precision=9)

    def set_time(self, time):
        self.kday = int(time.mjd // 1000) * 1000
        self.jday = int(time.mjd - self.kday)
        sec = (time - Time(self.kday+self.jday, format='mjd')).sec
        self.seconds, ns = divmod(sec, 1)
        self.ns = round(ns * 1e9)

    time = property(get_time, set_time)
