"""
Definitions for VLBI Mark5B Headers.

Implements a Mark5BHeader class used to store header words, and decode/encode
the information therein.

For the specification, see
http://www.haystack.edu/tech/vlbi/mark5/docs/Mark%205B%20users%20manual.pdf
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from astropy import units as u
from astropy.time import Time

from ..vlbi_base.header import HeaderParser, VLBIHeaderBase, four_word_struct
from ..vlbi_base.utils import bcd_decode, bcd_encode, CRC


__all__ = ['CRC16', 'crc16', 'Mark5BHeader']

CRC16 = 0x18005
"""CRC polynomial used for Mark 5B Headers, as a check on the time code.

x^16 + x^15 + x^2 + 1, i.e., 0x18005.
See page 11 of http://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf
(defined there for VLBA headers).

This is also CRC-16-IBM mentioned in
https://en.wikipedia.org/wiki/Cyclic_redundancy_check
"""
crc16 = CRC(CRC16)


class Mark5BHeader(VLBIHeaderBase):
    """Decoder/encoder of a Mark5B Frame Header.

    See page 15 of
    http://www.haystack.edu/tech/vlbi/mark5/docs/Mark%205B%20users%20manual.pdf

    Parameters
    ----------
    words : tuple of int, or None
        Eight (or four for legacy VDIF) 32-bit unsigned int header words.
        If ``None``, set to a tuple of zeros for later initialisation.
    kday : int, or None, optional
        Explicit thousands of MJD of the observation time (needed to remove
        ambiguity in the Mark 5B time stamp).  Can instead pass an approximate
        `ref_time`.
    ref_time : `~astropy.time.Time`, or None, optional
        Reference time within 500 days of the observation time, used to infer
        the full MJD.  Used only if `kday` is ``None``.
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
         ('crc', (3, 0, 16))))

    _struct = four_word_struct

    _properties = ('payloadsize', 'framesize', 'kday', 'jday', 'seconds',
                   'ns', 'time')
    """Properties accessible/usable in initialisation."""

    kday = None

    def __init__(self, words, kday=None, ref_time=None, verify=True, **kwargs):
        super(Mark5BHeader, self).__init__(words, verify=False, **kwargs)
        if kday is not None:
            self.kday = kday
        elif ref_time is not None:
            self.kday = self.infer_kday(ref_time, self.jday)
        if verify:
            self.verify()

    def verify(self):
        """Verify header integrity."""
        assert len(self.words) == 4
        assert (self['sync_pattern'] ==
                self._header_parser.defaults['sync_pattern'])
        assert self.kday is None or (33000 < self.kday < 400000)
        if self.kday is not None:
            assert self.kday % 1000 == 0, ("kday must be explicit "
                                           "thousands of MJD.")

    def copy(self, **kwargs):
        return super(Mark5BHeader, self).copy(kday=self.kday, **kwargs)

    def update(self, **kwargs):
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

        super(Mark5BHeader, self).update(**kwargs)
        if calculate_crc:
            # Do not use words 2 & 3 directly, so that this works also if part
            # of a VDIF header, where the time information is in words 7 & 8.
            stream = '{:012b}{:020b}{:016b}'.format(self['bcd_jday'],
                                                    self['bcd_seconds'],
                                                    self['bcd_fraction'])
            stream = np.array([int(b) for b in stream], dtype=np.uint8)
            crc = crc16(stream)
            self['crc'] = int(''.join(['{:1d}'.format(c) for c in crc]),
                              base=2)
            if verify:
                self.verify()

    @staticmethod
    def infer_kday(ref_time, header_jday):
        """Uses a reference time to determine a header's ``kday``.

        Parameters
        ----------
        ref_time : `~astropy.time.Time`
            Reference time within 500 days of the observation time.
        header_jday : int
            Correct jday from the header.

        Returns
        -------
        kday : int
            Explicit thousands of MJD of the observation time.
        """
        ref_kday, ref_jday = divmod(ref_time.mjd, 1000)
        return 1000 * int(ref_kday + np.round((ref_jday -
                                               header_jday) / 1000.))

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
        """Fractional seconds (in ns; decoded from 'bcd_fraction').

        The fraction is stored to 0.1 ms accuracy.  Following mark5access, this
        is "unrounded" to give the exact time of the start of the frame for any
        total bit rate below 512 Mbps.  For rates above this value, it is no
        longer guaranteed that subsequent frames have unique rates.

        Note to the above: since a Mark5B frame contains 80000 bits, the total
        bit rate for which times can be unique would in principle be 800 Mbps.
        However, standard VLBI only uses bit rates that are powers of 2 in MHz.
        """
        ns = bcd_decode(self['bcd_fraction']) * 100000
        # "unround" the nanoseconds
        return 156250 * ((ns + 156249) // 156250)

    @ns.setter
    def ns(self, ns):
        # From inspecting sample files, the fraction appears to be truncated,
        # not rounded.
        fraction = int(ns / 100000)
        self['bcd_fraction'] = bcd_encode(fraction)

    def get_time(self, framerate=None, frame_nr=None):
        """Convert year, BCD time code to Time object.

        Uses bcd-encoded 'jday', 'seconds', and 'frac_sec', plus ``kday``
        from the initialisation to calculate the time.  See
        http://www.haystack.edu/tech/vlbi/mark5/docs/Mark%205B%20users%20manual.pdf

        Note that some non-compliant files have 'frac_sec' not set.  For those,
        the time can still be retrieved using 'frame_nr' given a frame rate.

        Furthermore, fractional seconds are stored only to 0.1 ms accuracy.
        In the code, this is "unrounded" to give the exact time of the start
        of the frame for any total bit rate below 512 Mbps.  For rates above
        this value, it is no longer guaranteed that subsequent frames have
        unique rates, and one should pass in an explicit frame rate instead.

        Parameters
        ----------
        framerate : `~astropy.units.Quantity`, optional
            For non-zero `frame_nr`, this is used to calculate the
            corresponding offset.
        frame_nr : int, optional
            Can be used to override the ``frame_nr`` from the header.  If 0,
            the routine returns the time to integer seconds.

        Returns
        -------
        `~astropy.time.Time`
        """
        if framerate is None and frame_nr is None:
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
        return Time(self.kday + self.jday, (self.seconds + offset) / 86400,
                    format='mjd', scale='utc', precision=9)

    def set_time(self, time):
        self.kday = int(time.mjd // 1000) * 1000
        self.jday = int(time.mjd - self.kday)
        ns = int(round((time - Time(self.kday + self.jday, format='mjd')).sec *
                       1e9))
        sec, ns = divmod(ns, 1000000000)
        self.seconds = sec
        self.ns = ns

    time = property(get_time, set_time)
