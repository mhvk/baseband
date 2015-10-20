"""
Definitions for VLBI Mark 4 payloads.

Implements a Mark4Payload class used to store payload words, and decode to
or encode from a data array.

For the specification, see
http://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from ..vlbi_base.frame import VLBIFrameBase
from .header import Mark4Header, PAYLOADSIZE
from .payload import Mark4Payload


__all__ = ['Mark4Frame']

VALIDSTART = 160


class Mark4Frame(VLBIFrameBase):
    """Representation of a Mark 4 frame, consisting of a header and payload.

    Parameters
    ----------
    header : Mark4Header
        Wrapper around the encoded header words, providing access to the
        header information.
    payload : Mark4Payload
        Wrapper around the payload, provding mechanisms to decode it.
    verify : bool
        Whether or not to do basic assertions that check the integrity
        (e.g., that channel information and number of tracks are consistent
        between header and data).  Default: `True`.

    The Frame can also be read instantiated using class methods:

      fromfile : read header and payload from a filehandle

      fromdata : encode data as payload

    It also has methods to do the opposite:

      tofile : write header and payload to filehandle

      todata : decode payload to data

    A number of properties are defined: ``shape`` and ``dtype`` are the shape
    and type of the data array, ``words`` the full encoded frame, and ``size``
    the frame size in bytes.  Furthermore, the frame acts as a dictionary, with
    keys those of the header, and any attribute that is not defined on the
    frame itself, such as ``.time`` will be looked up on the header.
    """

    _header_class = Mark4Header
    _payload_class = Mark4Payload

    def __init__(self, header, payload, verify=True):
        self.header = header
        self.payload = payload
        if verify:
            self.verify()

    @property
    def valid(self):
        """Whether frame contains valid data.

        None of the error flags are set.
        """
        return not np.any(self.header['time_sync_error'] |
                          self.header['internal_clock_error'] |
                          self.header['processor_time_out_error'] |
                          self.header['communication_error'])

    @valid.setter
    def valid(self, valid):
        self.header['communication_error'] = ~valid

    @classmethod
    def fromfile(cls, fh, ntrack, decade=None, verify=True):
        """Read a frame from a filehandle.

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
        header = cls._header_class.fromfile(fh, ntrack, decade, verify)
        payload = cls._payload_class.fromfile(fh, header=header)
        return cls(header, payload, verify=verify)

    @classmethod
    def fromdata(cls, data, header=None, verify=True, **kwargs):
        """Construct frame from data and header.

        Parameters
        ----------
        data : ndarray
            Array holding complex or real data to be encoded.  This should have
            the full size of a data frame, even though the part covered by the
            header will be ignored.
        header : Mark4Header or None
            If `None`, it will be attemtped to create one using the keywords.
        verify : bool
            Whether or not to do basic assertions that check the integrity.
        """
        if header is None:
            header = cls._header_class.fromvalues(verify=verify, **kwargs)
        assert VALIDSTART * data.shape[0] % PAYLOADSIZE == 0
        start = VALIDSTART * data.shape[0] // PAYLOADSIZE
        payload = cls._payload_class.fromdata(data[start:], header=header)

        return cls(header, payload, verify=verify)

    def todata(self, data=None, invalid_data_value=0.):
        """Decode the payload.

        Parameters
        ----------
        data : None or ndarray
            If given, the data is decoded into the array (which should have
            the correct shape).  By default, a new array is created.
        invalid_data_value : float
            Value to use for part of frame overwritten by header as well as
            for the rest if the frame is invalid (default: 0.).
        """
        if data is None:
            data = np.empty(self.shape, self.dtype)
        if self.valid:
            valid_start = data.shape[0] * VALIDSTART // PAYLOADSIZE
            data[:valid_start] = invalid_data_value
            data[valid_start:] = self.payload.data
        else:
            data[...] = invalid_data_value
        return data

    data = property(todata, doc="Decode the payload.")

    @property
    def shape(self):
        """Shape of the data held in the payload (samples_per_frame, nchan)."""
        return (self.payload.shape[0] * PAYLOADSIZE //
                (PAYLOADSIZE - VALIDSTART), self.payload.shape[1])
