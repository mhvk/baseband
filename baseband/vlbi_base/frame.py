# Licensed under the GPLv3 - see LICENSE
"""
Base definitions for VLBI frames, used for VDIF and Mark 5B.

Defines a frame class VLBIFrameBase that can be used to hold a header and a
payload, providing access to the values encoded in both.
"""
# Helper functions for VLBI readers (VDIF, Mark5B).
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from astropy.extern import six

__all__ = ['VLBIFrameBase']


class VLBIFrameBase(object):
    """Representation of a VLBI data frame, consisting of a header and payload.

    Parameters
    ----------
    header : `baseband.vlbi_base.header.VLBIHeaderBase`
        Wrapper around the encoded header words, providing access to the
        header information.
    payload : `~baseband.vlbi_base.payload.VLBIPayloadBase`
        Wrapper around the payload, provding mechanisms to decode it.
    valid : bool
        Whether the data are valid.  Default: `True`.
    verify : bool
        Whether to do basic verification of integrity.  Default: `True`.

    Notes
    -----
    The Frame can also be instantiated using class methods:

      fromfile : read header and payload from a filehandle

      fromdata : encode data as payload

    Of course, one can also do the opposite:

      tofile : method to write header and payload to filehandle

      data : property that yields full decoded payload

    One can decode part of the payload by indexing or slicing the frame.
    If the frame does not contain valid data, all values returned are set
    to ``self.fill_value``.

    A number of properties are defined: `shape` and `dtype` are the shape
    and type of the data array, and `nbytes` the frame size in bytes.
    Furthermore, the frame acts as a dictionary, with keys those of the header.
    Any attribute that is not defined on the frame itself, such as ``.time``
    will be looked up on the header as well.
    """

    _header_class = None
    _payload_class = None
    _fill_value = 0.

    def __init__(self, header, payload, valid=True, verify=True):
        self.header = header
        self.payload = payload
        self.valid = valid
        if verify:
            self.verify()

    def verify(self):
        """Simple verification.  To be added to by subclasses."""
        assert isinstance(self.header, self._header_class)
        assert isinstance(self.payload, self._payload_class)
        assert (self.payload.nbytes ==
                self.payload.words.size * self.payload.words.dtype.itemsize)
        assert (self.payload.nbytes == getattr(self.header, 'payload_nbytes',
                                               self.payload.nbytes))

    @property
    def valid(self):
        """Whether frame contains valid data."""
        return self._valid

    @valid.setter
    def valid(self, valid):
        self._valid = bool(valid)

    @classmethod
    def fromfile(cls, fh, *args, **kwargs):
        """Read a frame from a filehandle.

        Any arguments beyond the filehandle are used to help initialize the
        payload, except for ``valid`` and ``verify``, which are passed on to
        the header and class initializers.
        """
        valid = kwargs.pop('valid', True)
        verify = kwargs.pop('verify', True)
        header = cls._header_class.fromfile(fh, verify=verify)
        payload = cls._payload_class.fromfile(fh, *args, **kwargs)
        return cls(header, payload, valid=valid, verify=verify)

    def tofile(self, fh):
        """Write encoded frame to filehandle."""
        self.header.tofile(fh)
        self.payload.tofile(fh)

    @classmethod
    def fromdata(cls, data, header, *args, **kwargs):
        """Construct frame from data and header.

        Parameters
        ----------
        data : `~numpy.ndarray`
            Array holding data to be encoded.
        header : `~baseband.vlbi_base.header.VLBIHeaderBase`
            Header for the frame.
        *args, **kwargs :
            Any arguments beyond the filehandle are used to help initialize the
            payload, except for ``valid`` and ``verify``, which are passed on
            to the header and class initializers.
        """
        valid = kwargs.pop('valid', True)
        verify = kwargs.pop('verify', True)
        payload = cls._payload_class.fromdata(data, *args, **kwargs)
        return cls(header, payload, valid=valid, verify=verify)

    @property
    def sample_shape(self):
        """Shape of a sample in the frame (nchan,)."""
        return self.payload.sample_shape

    def __len__(self):
        """Number of samples in the frame."""
        return len(self.payload)

    @property
    def shape(self):
        """Shape of the frame data."""
        return (len(self),) + self.sample_shape

    @property
    def size(self):
        """Total number of component samples in the frame data."""
        prod = 1
        for dim in self.shape:
            prod *= dim
        return prod

    @property
    def ndim(self):
        """Number of dimensions of the frame data."""
        return len(self.shape)

    @property
    def dtype(self):
        """Numeric type of the frame data."""
        return self.payload.dtype

    @property
    def nbytes(self):
        """Size of the encoded frame in bytes."""
        return self.header.nbytes + self.payload.nbytes

    @property
    def fill_value(self):
        """Value to replace invalid data in the frame."""
        return self._fill_value

    @fill_value.setter
    def fill_value(self, fill_value):
        self._fill_value = fill_value

    def __array__(self, dtype=None):
        """Interface to arrays."""
        if dtype is None or dtype == self.dtype:
            return self.data
        else:
            return self.data.astype(dtype)

    # Header behaves as a dictionary, while Payload can be indexed/sliced.
    # Let frame behave appropriately.
    def __getitem__(self, item=()):
        if isinstance(item, six.string_types):
            return self.header.__getitem__(item)
        elif self.valid:
            return self.payload.__getitem__(item)
        else:
            data_shape = np.empty(self.shape, dtype=bool)[item].shape
            return np.full(data_shape, self.fill_value,
                           dtype=self.dtype)

    data = property(__getitem__, doc="Full decoded frame.")

    def __setitem__(self, item, value):
        if isinstance(item, six.string_types):
            self.header.__setitem__(item, value)
        else:
            self.payload.__setitem__(item, value)

    def keys(self):
        return self.header.keys()

    def _ipython_key_completions_(self):
        # Enables tab-completion of header keys in IPython.
        return self.header.keys()

    def __contains__(self, key):
        return key in self.keys()

    # Try to get any attribute not on the frame from the header properties.
    def __getattr__(self, attr):
        if attr in self.header._properties:
            return getattr(self.header, attr)
        else:
            # Raise appropriate error.
            return self.__getattribute__(attr)

    # For tests, it is useful to define equality.
    def __eq__(self, other):
        return (type(self) is type(other) and
                self.valid == other.valid and
                self.header == other.header and
                self.payload == other.payload)
