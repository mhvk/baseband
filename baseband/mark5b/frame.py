# Licensed under the GPLv3 - see LICENSE
"""
Definitions for VLBI Mark 5B frames.

Implements a Mark5BFrame class that can be used to hold a header and a
payload, providing access to the values encoded in both.

For the specification, see
http://www.haystack.edu/tech/vlbi/mark5/docs/Mark%205B%20users%20manual.pdf
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from ..vlbi_base.frame import VLBIFrameBase
from .header import Mark5BHeader
from .payload import Mark5BPayload


__all__ = ['Mark5BFrame']


class Mark5BFrame(VLBIFrameBase):
    """Representation of a Mark 5B frame, consisting of a header and payload.

    Parameters
    ----------
    header : `~baseband.mark5b.Mark5BHeader`
        Wrapper around the encoded header words, providing access to the
        header information.
    payload : `~baseband.mark5b.Mark5BPayload`
        Wrapper around the payload, provding mechanisms to decode it.
    valid : bool or None
        Whether the data are valid.  If `None` (default), the validity will be
        determined by checking whether the payload consists of the fill pattern
        0x11223344.
    verify : bool
        Whether to do basic verification of integrity (default: `True`)

    Notes
    -----
    The Frame can also be read instantiated using class methods:

      fromfile : read header and payload from a filehandle

      fromdata : encode data as payload

    Of course, one can also do the opposite:

      tofile : method to write header and payload to filehandle

      data : property that yields full decoded payload

    A number of properties are defined: `shape`, `dtype` and `size` are
    the shape, type and number of complete samples of the data array, and
    `nbytes` the frame size in bytes.  Furthermore, the frame acts as a
    dictionary, with keys those of the header.  Any attribute that is not
    defined on the frame itself, such as ``.time`` will be looked up on the
    header as well.
    """

    _header_class = Mark5BHeader
    _payload_class = Mark5BPayload
    _fill_pattern = 0x11223344

    def __init__(self, header, payload, valid=None, verify=True):
        if valid is None:
            # Is this payload OK?  Usually yes, so short-circuit on first few.
            valid = (payload.words[0] != self._fill_pattern or
                     payload.words[1] != self._fill_pattern or
                     payload.words[2] != self._fill_pattern or
                     (payload.words[3:] != self._fill_pattern).any())

        super(Mark5BFrame, self).__init__(header, payload, valid, verify)

    @classmethod
    def fromfile(cls, fh, kday=None, ref_time=None, nchan=1, bps=3, valid=None,
                 verify=True):
        """Read a frame from a filehandle.

        Parameters
        ----------
        fh : filehandle
            To read the header and payload from.
        kday : int or None
            Explicit thousands of MJD of the observation time.  Can instead
            pass an approximate ``ref_time``.
        ref_time : `~astropy.time.Time` or None
            Reference time within 500 days of the observation time, used to
            infer the full MJD.  Used only if ``kday`` is not given.
        nchan : int, optional
            Number of channels.   Default: 1.
        bps : int, optional
            Bits per elementary sample.  Default: 2.
        verify : bool
            Whether to do basic checks of frame integrity (default: `True`).
        """
        header = cls._header_class.fromfile(fh, kday=kday, ref_time=ref_time,
                                            verify=verify)
        payload = cls._payload_class.fromfile(fh, nchan=nchan, bps=bps)
        return cls(header, payload, valid, verify)

    @classmethod
    def fromdata(cls, data, header=None, bps=2, valid=True, verify=True,
                 **kwargs):
        """Construct frame from data and header.

        Parameters
        ----------
        data : `~numpy.ndarray`
            Array holding data to be encoded.
        header : `~baseband.mark5b.Mark5BHeader` or None
            If not given, will attempt to generate one using the keywords.
        bps : int
            Bits per elementary sample.  Default: 2.
        valid : bool
            Whether the data are valid (default: `True`).  If not, the payload
            will be set to a fill pattern.
        verify : bool
            Whether to do basic checks of frame integrity (default: `True`).
        """
        if header is None:
            header = Mark5BHeader.fromvalues(verify=verify, **kwargs)
        payload = cls._payload_class.fromdata(data, bps=bps)
        return cls(header, payload, valid=valid, verify=verify)

    def tofile(self, fh):
        """Write encoded frame to filehandle."""
        self.header.tofile(fh)
        if self.valid:
            self.payload.tofile(fh)
        else:
            fh.write(np.full_like(self.payload.words,
                                  self._fill_pattern).tostring())
