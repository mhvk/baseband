# Licensed under the GPLv3 - see LICENSE
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from astropy.extern import six

from ..vlbi_base.frame import VLBIFrameBase
from .header import GUPPIHeader
from .payload import GUPPIPayload


__all__ = ['GUPPIFrame']


class GUPPIFrame(VLBIFrameBase):
    """Representation of a GUPPI file, consisting of a header and payload.

    Parameters
    ----------
    header : `~baseband.dada.GUPPIHeader`
        Wrapper around the header lines, providing access to the values.
    payload : `~baseband.dada.GUPPIPayload`
        Wrapper around the payload, provding mechanisms to decode it.
    valid : bool, optional
        Whether the data are valid.  Default: `True`.
    verify : bool, optional
        Whether to do basic verification of integrity.  Default: `True`.

    Notes
    -----
    GUPPI files do not support storing whether data are valid or not on disk.
    Hence, this has to be determined independently.  If ``valid=False``, any
    decoded data are set to ``cls.fill_value`` (by default, 0).

    The Frame can also be instantiated using class methods:

      fromfile : read header and and map or read payload from a filehandle

      fromdata : encode data as payload

    Of course, one can also do the opposite:

      tofile : method to write header and payload to filehandle

      data : property that yields full decoded payload

    One can decode part of the payload by indexing or slicing the frame.

    A number of properties are defined: `shape`, `dtype` and `size` are
    the shape, type and number of complete samples of the data array, and
    `nbytes` the frame size in bytes.  Furthermore, the frame acts as a
    dictionary, with keys those of the header.  Any attribute that is not
    defined on the frame itself, such as ``.time`` will be looked up on the
    header as well.
    """
    _header_class = GUPPIHeader
    _payload_class = GUPPIPayload

    @classmethod
    def fromfile(cls, fh, memmap=True, valid=True, verify=True):
        """Read a frame from a filehandle, possible mapping the payload.

        Parameters
        ----------
        fh : filehandle
            To read header from.
        memmap : bool, optional
            If `True` (default), use `~numpy.memmap` to map the payload.
            If `False`, just read it from disk.
        valid : bool, optional
            Whether the data are valid (default: `True`). Note that this cannot
            be inferred from the header or payload itself.  If `False`, any
            data read will be set to ``cls.fill_value``.
        verify : bool, optional
            Whether to do basic verification of integrity.  Default: `True`.
        """
        header = cls._header_class.fromfile(fh, verify=verify)
        payload = cls._payload_class.fromfile(fh, header=header, memmap=memmap)
        return cls(header, payload, valid=valid, verify=verify)

    @classmethod
    def fromdata(cls, data, header=None, valid=True, verify=True, **kwargs):
        """Construct frame from data and header.

        Note that since GUPPI files are generally very large, one would
        normally map the file, and then set pieces of it by assigning to slices
        of the frame.  See `~baseband.guppi.base.GUPPIFileWriter.memmap_frame`.

        Parameters
        ----------
        data : `~numpy.ndarray`
            Array holding complex or real data to be encoded.
        header : `~baseband.dada.GUPPIHeader` or None, optional
            If not given, will attempt to generate one using the keywords.
        valid : bool, optional
            Whether the data are valid (default: `True`). Note that this
            information cannot be written to disk.
        verify : bool, optional
            Whether or not to do basic assertions that check the integrity.
            Default: `True`.
        **kwargs
            If ``header`` is not given, these are used to initialize one.
        """
        if header is None:
            header = cls._header_class.fromvalues(verify=verify, **kwargs)
        payload = cls._payload_class.fromdata(data, header=header)
        return cls(header, payload, valid=valid, verify=verify)

    def __len__(self):
        """Number of samples, subtracting those in the overlap region."""
        return self.header.samples_per_frame

    def _get_payload_item(self, item):
        """Translate frame item to payload item, preventing access to the
        overlap section at the end of the payload.

        Parameters
        ----------
        item : int, slice, or tuple
            Sample indices.  An int represents a single sample, a slice
            a sample range, and a tuple of ints/slices a range for
            multi-channel data.

        Returns
        -------
        payload_item : int, slice, or None
            Part of the payload that should be decoded.
        sample_index : tuple
            Any slicing beyond the sample number.

        Notes
        -----
        ``item`` is restricted to (tuples of) ints or slices, so one cannot
        access non-contiguous samples using advanced indexing.  If ``item``
        is a slice, a negative increment cannot be used.
        """
        nsample = len(self)
        if item is () or item == slice(None):
            # Short-cut for full payload.
            return slice(nsample), ()

        if isinstance(item, tuple):
            sample_index = item[1:]
            item = item[0]
        else:
            sample_index = ()

        if isinstance(item, slice):
            # Translate slice to an array of nsample length.
            payload_item = slice(*item.indices(nsample))
        else:
            # Not a slice. Maybe an index?
            try:
                item = item.__index__()
            except Exception:
                raise TypeError("{0} object can only be indexed or sliced."
                                .format(type(self)))
            if item < 0:
                item += nsample

            if item > nsample:
                raise IndexError("{0} index out of range.".format(type(self)))

        return payload_item, sample_index

    def __getitem__(self, item=()):
        if isinstance(item, six.string_types):
            return self.header.__getitem__(item)
        elif not self.valid:
            data_shape = np.empty(self.shape, dtype=bool)[item].shape
            return np.full(data_shape, self.fill_value,
                           dtype=self.dtype)
        # Normally, we would just pass on to the payload here, but for
        # GUPPI, each frame ends with an "overlap section" of samples identical
        # to the first samples in the next frame.  We do not decode these.
        payload_item, sample_index = self._get_payload_item(item)
        data = self.payload[payload_item]
        if sample_index is ():
            return data
        else:
            return data[(Ellipsis,) + sample_index]

    def __setitem__(self, item, value):
        if isinstance(item, six.string_types):
            return self.header.__setitem__(item, value)

        # Normally, we would just pass on to the payload here, but for
        # GUPPI we must handle the "overlap section".
        data = np.asanyarray(value)
        assert data.ndim <= 3

        payload_item, sample_index = self._get_payload_item(item)

        if sample_index is not ():
            payload_item = (payload_item,) + sample_index

        self.payload[payload_item] = data

    data = property(__getitem__,
                    doc="Full decoded frame, except for the overlap section.")
