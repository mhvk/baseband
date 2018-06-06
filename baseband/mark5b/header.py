# Licensed under the GPLv3 - see LICENSE
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
import astropy.units as u
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
        Four 32-bit unsigned int header words.  If `None`, set to a tuple of
        zeros for later initialisation.
    kday : int or None
        Explicit thousands of MJD of the observation time (needed to remove
        ambiguity in the Mark 5B time stamp).  Can instead pass an approximate
        ``ref_time``.
    ref_time : `~astropy.time.Time` or None
        Reference time within 500 days of the observation time, used to infer
        the full MJD.  Used only if ``kday`` is not given.
    verify : bool, optional
        Whether to do basic verification of integrity.  Default: `True`.

    Returns
    -------
    header : `Mark5BHeader`
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
    _sync_pattern = _header_parser.defaults['sync_pattern']

    _struct = four_word_struct

    _properties = ('payload_nbytes', 'frame_nbytes', 'kday', 'jday', 'seconds',
                   'fraction', 'time')
    """Properties accessible/usable in initialisation."""

    kday = None
    _payload_nbytes = 10000  # 2500 words

    def __init__(self, words, kday=None, ref_time=None, verify=True, **kwargs):
        super(Mark5BHeader, self).__init__(words, verify=False, **kwargs)
        if kday is not None:
            self.kday = kday
        elif ref_time is not None:
            self.infer_kday(ref_time)
        if verify:
            self.verify()

    def verify(self):
        """Verify header integrity."""
        assert len(self.words) == 4
        assert self['sync_pattern'] == self._sync_pattern
        assert self.kday is None or (33000 < self.kday < 400000)
        if self.kday is not None:
            assert self.kday % 1000 == 0, "kday must be thousands of MJD."

    def copy(self, **kwargs):
        return super(Mark5BHeader, self).copy(kday=self.kday, **kwargs)

    @classmethod
    def fromvalues(cls, **kwargs):
        """Initialise a header from parsed values.

        Here, the parsed values must be given as keyword arguments, i.e., for
        any ``header = cls(<data>)``, ``cls.fromvalues(**header) == header``.

        However, unlike for the :meth:`Mark5BHeader.fromkeys` class method,
        data can also be set using arguments named after methods, such as
        ``jday`` and ``seconds``.

        Given defaults:

        sync_pattern : 0xABADDEED

        Values set by other keyword arguments (if present):

        bcd_jday : from ``jday`` or ``time``
        bcd_seconds : from ``seconds`` or ``time``
        bcd_fraction : from ``fraction`` or ``time`` (may need ``frame_rate``)
        frame_nr : from ``time`` (may need ``frame_rate``)
        """
        time = kwargs.pop('time', None)
        frame_rate = kwargs.pop('frame_rate', None)
        # Pop verify and pass on False so verify happens after time is set.
        verify = kwargs.pop('verify', True)
        self = super(Mark5BHeader, cls).fromvalues(verify=False, **kwargs)
        if time is not None:
            self.set_time(time, frame_rate=frame_rate)
            self.update()    # Recalculate CRC.
        if verify:
            self.verify()
        return self

    def update(self, **kwargs):
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

    def infer_kday(self, ref_time):
        """Uses a reference time to set a header's ``kday``.

        Parameters
        ----------
        ref_time : `~astropy.time.Time`
            Reference time within 500 days of the observation time.
        """
        self.kday = np.round(ref_time.mjd - self.jday, decimals=-3).astype(int)

    @property
    def payload_nbytes(self):
        """Size of the payload in bytes."""
        return self._payload_nbytes    # Hardcoded in class definition.

    @payload_nbytes.setter
    def payload_nbytes(self, payload_nbytes):
        if payload_nbytes != self._payload_nbytes:  # 2500 words.
            raise ValueError("Mark 5B payload has a fixed size of 10000 bytes "
                             "(2500 words).")

    @property
    def frame_nbytes(self):
        """Size of the frame in bytes."""
        return self.nbytes + self.payload_nbytes

    @frame_nbytes.setter
    def frame_nbytes(self, frame_nbytes):
        if frame_nbytes != self.nbytes + self.payload_nbytes:
            raise ValueError("Mark 5B frame has a fixed size of 10016 bytes "
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
    def fraction(self):
        """Fractional seconds (decoded from 'bcd_fraction').

        The fraction is stored to 0.1 ms accuracy.  Following mark5access, this
        is "unrounded" to give the exact time of the start of the frame for any
        total bit rate below 512 Mbps.  For rates above this value, it is no
        longer guaranteed that subsequent frames have unique rates.

        Note to the above: since a Mark5B frame contains 80000 bits, the total
        bit rate for which times can be unique would in principle be 800 Mbps.
        However, standard VLBI only uses bit rates that are powers of 2 in MHz.
        """
        ns = bcd_decode(self['bcd_fraction']) * 100000
        # "Unround" the nanoseconds, and turn into fractional seconds.
        return (156250 * ((ns + 156249) // 156250)) / 1e9

    @fraction.setter
    def fraction(self, fraction):
        ns = round(fraction * 1.e9)
        # From inspecting sample files, the fraction appears to be truncated,
        # not rounded.
        fraction = int(ns / 100000)
        self['bcd_fraction'] = bcd_encode(fraction)

    def get_time(self, frame_rate=None):
        """Convert year, BCD time code to Time object.

        Calculate time using `jday`, `seconds`, and `fraction` properties
        (which reflect the bcd-encoded 'bcd_jday', 'bcd_seconds' and
        'bcd_fraction' header items), plus `kday` from the initialisation.  See
        http://www.haystack.edu/tech/vlbi/mark5/docs/Mark%205B%20users%20manual.pdf

        Note that some non-compliant files do not have 'bcd_fraction' set.
        For those, the time can still be calculated using the header's
        'frame_nr' by passing in a frame rate.

        Furthermore, fractional seconds are stored only to 0.1 ms accuracy.
        In the code, this is "unrounded" to give the exact time of the start
        of the frame for any total bit rate below 512 Mbps.  For higher rates,
        it is no longer guaranteed that subsequent frames have unique
        `fraction`, and one should pass in an explicit frame rate instead.

        Parameters
        ----------
        frame_rate : `~astropy.units.Quantity`, optional
            Used to calculate the fractional second from the frame number
            instead of from the header's `fraction`.

        Returns
        -------
        `~astropy.time.Time`
        """
        frame_nr = self['frame_nr']
        if frame_nr == 0:
            fraction = 0.
        elif frame_rate is None:
            fraction = self.fraction
            if fraction == 0.:
                raise ValueError('header does not provide correct fractional '
                                 'second (it is zero for non-zero frame '
                                 'number). Please pass in a frame_rate.')
        else:
            fraction = (frame_nr / frame_rate).to_value(u.s)

        return Time(self.kday + self.jday, (self.seconds + fraction) / 86400,
                    format='mjd', scale='utc', precision=9)

    def set_time(self, time, frame_rate=None):
        """
        Convert Time object to BCD timestamp elements and 'frame_nr'.

        For non-integer seconds, the frame number will be calculated if not
        given explicitly. Doing so requires the frame rate.

        Parameters
        ----------
        time : `~astropy.time.Time`
            The time to use for this header.
        frame_rate : `~astropy.units.Quantity`, optional
            For calculating 'frame_nr' from the fractional seconds.
        """
        self.kday = int(time.mjd // 1000) * 1000
        self.jday = int(time.mjd - self.kday)
        seconds = time - Time(self.kday + self.jday, format='mjd')
        int_sec = int(seconds.sec)
        fraction = seconds - int_sec * u.s

        # Round to nearest ns to handle timestamp difference errors.
        if abs(fraction) < 1. * u.ns:
            frame_nr = 0
            frac_sec = 0.
        elif abs(1. * u.s - fraction) < 1. * u.ns:
            int_sec += 1
            frame_nr = 0
            frac_sec = 0.
        else:
            if frame_rate is None:
                raise ValueError("cannot calculate frame rate. Pass it "
                                 "in explicitly.")
            frame_nr = int(round((fraction * frame_rate).to(u.one).value))
            fraction = frame_nr / frame_rate
            if abs(fraction - 1. * u.s) < 1. * u.ns:
                int_sec += 1
                frame_nr = 0
                frac_sec = 0.
            else:
                frac_sec = fraction.to(u.s).value

        self.seconds = int_sec
        self.fraction = frac_sec
        self['frame_nr'] = frame_nr

    time = property(get_time, set_time)
