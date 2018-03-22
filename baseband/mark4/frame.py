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
from astropy.extern import six

from ..vlbi_base.frame import VLBIFrameBase
from .header import Mark4Header
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
    valid : bool or `None`
        Whether the data is valid.  If `None` (default), inferred from header.
        Note that the header is updated in-place if `True` or `False`.
    verify : bool
        Whether or not to do basic assertions that check the integrity
        (e.g., that channel information and number of tracks are consistent
        between header and data).  Default: `True`.

    Notes
    -----

    The Frame can also be read instantiated using class methods:

      fromfile : read header and payload from a filehandle

      fromdata : encode data as payload

    Of course, one can also do the opposite:

      tofile : method to write header and payload to filehandle

      data : property that yields full decoded payload

    One can decode part of the payload by indexing or slicing the frame.
    If the frame does not contain valid data, all values returned are set
    to ``self.invalid_data_value``.

    A number of properties are defined: ``shape`` and ``dtype`` are the shape
    and type of the data array, ``words`` the full encoded frame, and ``size``
    the frame size in bytes.  Furthermore, the frame acts as a dictionary, with
    keys those of the header. Any attribute that is not defined on the frame
    itself, such as ``.time`` will be looked up on the header as well.
    """

    _header_class = Mark4Header
    _payload_class = Mark4Payload

    def __init__(self, header, payload, valid=None, verify=True):
        self.header = header
        self.payload = payload
        if valid is not None:
            self.valid = valid
        if verify:
            self.verify()
        self._valid_start = VALIDSTART * header.fanout

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
        if valid:
            self.header['time_sync_error'] = False
            self.header['internal_clock_error'] = False
            self.header['processor_time_out_error'] = False
            self.header['communication_error'] = False
        else:
            self.header['communication_error'] = True

    @classmethod
    def fromfile(cls, fh, ntrack, decade=None, ref_time=None, verify=True):
        """Read a frame from a filehandle.

        Parameters
        ----------
        fh : filehandle
            To read header from.
        ntrack : int
            Number of Mark 4 bitstreams.
        decade : int, or None, optional
            Decade in which the observations were taken.  Can instead pass an
            approximate `ref_time`.
        ref_time : `~astropy.time.Time`, or None, optional
            Reference time within 4 years of the observation time.  Used only
            if `decade` is ``None``.

        verify : bool
            Whether to do basic verification of integrity.  Default: `True`.
        """
        header = cls._header_class.fromfile(fh, ntrack, decade=decade,
                                            ref_time=ref_time, verify=verify)
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
        assert data.shape[0] == header.samples_per_frame
        start = VALIDSTART * header.fanout
        payload = cls._payload_class.fromdata(data[start:], header=header)

        return cls(header, payload, verify=verify)

    @property
    def shape(self):
        """Shape of the data held in the payload (samples_per_frame, nchan)."""
        payload_shape = self.payload.shape
        return (self._valid_start + payload_shape[0],) + payload_shape[1:]

    def __getitem__(self, item=()):
        if isinstance(item, six.string_types):
            return self.header.__getitem__(item)

        # Normally, we would just pass on to the payload here, but for
        # Mark 4, we need to deal with data overwritten by the header.
        valid_start = self._valid_start
        if item is () or item == slice(None):
            # Short-cut for full payload.
            if self.valid:
                # Note: Creating an empty array and setting part to invalid
                # is much faster than creating one pre-filled with invalid.
                data = np.empty(self.shape, self.dtype)
                data[:valid_start] = self.invalid_data_value
                data[valid_start:] = self.payload.data
                return data
            else:
                return np.full(self.shape, self.invalid_data_value, self.dtype)

        if isinstance(item, tuple):
            sample_index = item[1:]
            item = item[0]
        else:
            sample_index = None

        # Interpret item as an index or slice.
        nsample = self.shape[0]
        if not isinstance(item, slice):
            try:
                item = item.__index__()
            except Exception:
                raise TypeError("{0} object can only be indexed or sliced."
                                .format(type(self)))
            if item < 0:
                item += nsample

            if not (0 <= item < nsample):
                raise IndexError("{0} index out of range.".format(type(self)))

            payload_item = item - valid_start
            if payload_item >= 0:
                data = self.payload[payload_item]
            else:
                data = np.full(self.sample_shape, self.invalid_data_value,
                               self.dtype)

            return data if sample_index is None else data[sample_index]

        # We have a slice.
        start, stop, step = item.indices(nsample)
        assert step > 0, "cannot deal with negative steps yet"
        
        payload_start = start - valid_start
        payload_stop = stop - valid_start
        if payload_start >= 0 and self.valid:
            # All requested data falls within the payload.
            data = self.payload[payload_start:payload_stop:step]
        else:
            shape = ((stop - start - 1) // step + 1,) + self.shape[1:]
            if payload_stop > 0 and self.valid:
                # Some requested data overlaps with the payload and is valid.
                data = np.empty(shape, self.dtype)
                ninvalid, payload_start = divmod(payload_start, step)
                ninvalid = -ninvalid
                data[:ninvalid] = self.invalid_data_value
                data[ninvalid:] = self.payload[payload_start:payload_stop:step]
            else:
                # Everything is invalid.
                data = np.full(shape, self.invalid_data_value, self.dtype)

        return data if sample_index is None else data[(slice(None),) +
                                                      sample_index]

    data = property(__getitem__,
                    doc="Decode the payload, invalidating the header part")
