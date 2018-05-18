# Licensed under the GPLv3 - see LICENSE
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
import operator
from astropy.extern import six

from ..vlbi_base.frame import VLBIFrameBase
from .header import Mark4Header
from .payload import Mark4Payload


__all__ = ['Mark4Frame']


class Mark4Frame(VLBIFrameBase):
    """Representation of a Mark 4 frame, consisting of a header and payload.

    Parameters
    ----------
    header : `~baseband.mark4.Mark4Header`
        Wrapper around the encoded header words, providing access to the
        header information.
    payload : `~baseband.mark4.Mark4Payload`
        Wrapper around the payload, provding mechanisms to decode it.
    valid : bool or None, optional
        Whether the data are valid.  If `None` (default), inferred from header.
        Note that `header` is updated in-place if `True` or `False`.
    verify : bool, optional
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
    to ``self.fill_value``.

    A number of properties are defined: `shape`, `dtype` and `size` are
    the shape, type and number of complete samples of the data array, and
    `nbytes` the frame size in bytes.  Furthermore, the frame acts as a
    dictionary, with keys those of the header.  Any attribute that is not
    defined on the frame itself, such as ``.time`` will be looked up on the
    header as well.
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
        decade : int or None
            Decade in which the observations were taken.  Can instead pass an
            approximate ``ref_time``.
        ref_time : `~astropy.time.Time` or None
            Reference time within 4 years of the observation time.  Used only
            if ``decade`` is not given.
        verify : bool, optional
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
        data : `~numpy.ndarray`
            Array holding complex or real data to be encoded.  This should have
            the full size of a data frame, even though the part covered by the
            header will be ignored.
        header : `~baseband.mark4.Mark4Header` or None
            If not given, will attempt to generate one using the keywords.
        verify : bool, optional
            Whether to do basic checks of frame integrity (default: `True`).
        """
        if header is None:
            header = cls._header_class.fromvalues(verify=verify, **kwargs)
        assert data.shape[0] == header.samples_per_frame
        # Start of part not overwritten by header
        # (see calculation of header.samples_per_frame)
        start = header.nbytes * 8 // (header.ntrack // header.fanout)
        payload = cls._payload_class.fromdata(data[start:], header=header)

        return cls(header, payload, verify=verify)

    def __len__(self):
        """Number of samples (including those overwritten by header)."""
        return self.header.samples_per_frame

    def _get_payload_item(self, item):
        """Translate frame item to payload item, correcting for header part.

        For Mark 4 frames, part of the actual data is overwritten by the
        header.  In slicing a frame, these parts should be set to invalid.

        Parameters
        ----------
        item : int, slice, or tuple
            Sample indices.  An int represents a single sample, a slice
            a sample range, and a tuple of ints/slices a range for
            multi-channel data.

        Returns
        -------
        payload_item : int, slice, or None
            Part of the payload that should be decoded. `None` if all the
            requested data is in the invalid part.
        sample_index : tuple
            Any slicing beyond the sample number.
        data_shape : tuple
            Shape of the data array that the indexing will lead to.  Useful
            to create an array that partially needs to be set to invalid.
        ninvalid : int
            Number of points in the data that should be set to invalid (or that
            should be ignored for setting).

        Notes
        -----
        ``item`` is restricted to (tuples of) ints or slices, so one cannot
        access non-contiguous samples using advanced indexing.  If ``item``
        is a slice, a negative increment cannot be used.
        """
        nsample = len(self)
        valid_start = nsample - len(self.payload)
        if item is () or item == slice(None):
            # Short-cut for full payload.
            return slice(None), (), self.shape, valid_start

        if isinstance(item, tuple):
            sample_index = item[1:]
            item = item[0]
        else:
            sample_index = ()

        if isinstance(item, slice):
            start, stop, step = item.indices(nsample)

            data_shape = ((stop - start - 1) // step + 1,) + self.sample_shape
            payload_start = start - valid_start
            payload_stop = stop - valid_start
            if payload_start >= 0:
                # All requested data falls within the payload.
                payload_item = slice(payload_start, payload_stop, step)
                ninvalid = 0
            elif payload_stop > 0:
                # Some data overlaps with the payload.
                ninvalid, payload_start = divmod(payload_start, step)
                ninvalid = -ninvalid
                payload_item = slice(payload_start, payload_stop, step)
            else:
                # Everything is invalid.
                payload_item = None
                ninvalid = data_shape[0]
        else:
            # Not a slice. Maybe an index?
            try:
                item = operator.index(item)
            except Exception:
                raise TypeError("{0} object can only be indexed or sliced."
                                .format(type(self)))
            if item < 0:
                item += nsample

            if not (0 <= item < nsample):
                raise IndexError("{0} index out of range.".format(type(self)))

            data_shape = self.sample_shape
            payload_item = item - valid_start
            if payload_item >= 0:
                ninvalid = 0
            else:
                payload_item = None
                ninvalid = 1

        return payload_item, sample_index, data_shape, ninvalid

    def __getitem__(self, item=()):
        if isinstance(item, six.string_types):
            return self.header.__getitem__(item)

        # Normally, we would just pass on to the payload here, but for
        # Mark 4, we need to deal with data overwritten by the header.
        (payload_item, sample_index, data_shape,
         ninvalid) = self._get_payload_item(item)
        if not self.valid or payload_item is None:
            data = np.full(data_shape, self.fill_value, self.dtype)

        elif ninvalid == 0:
            data = self.payload[payload_item]

        else:
            # Note: Creating an empty array and setting part to invalid
            # is much faster than creating one pre-filled with invalid.
            data = np.empty(data_shape, self.dtype)
            data[:ninvalid] = self.fill_value
            data[ninvalid:] = self.payload[payload_item]

        if sample_index is ():
            return data
        else:
            return data[(Ellipsis,) + sample_index]

    def __setitem__(self, item, value):
        if isinstance(item, six.string_types):
            return self.header.__setitem__(item, value)

        # Normally, we would just pass on to the payload here, but for
        # Mark 4, we need to deal with data overwritten by the header.
        data = np.asanyarray(value)
        assert data.ndim <= 2

        (payload_item, sample_index, data_shape,
         ninvalid) = self._get_payload_item(item)

        if payload_item is None:
            return

        if ninvalid > 0:
            # See if data has enough dimensions so that we need to remove
            # the part that cannot set anything in the payload.
            if sample_index is ():
                sample_ndim = len(self.sample_shape)
            else:
                sample_ndim = np.empty(self.sample_shape)[sample_index].ndim
            if data.ndim == 1 + sample_ndim:
                data = data[ninvalid:]

        if sample_index is not ():
            payload_item = (payload_item,) + sample_index

        self.payload[payload_item] = data

    data = property(__getitem__,
                    doc="Full decoded frame, with header part filled in.")
