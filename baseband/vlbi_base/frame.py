"""
Base definitions for VLBI frames, used for VDIF and Mark 5B.

Defines a frame class VLBIFrameBase that can be used to hold a header and a
payload, providing access to the values encoded in both.
"""
# Helper functions for VLBI readers (VDIF, Mark5B).
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from astropy.extern import six

__all__ = ['VLBIFrameBase']


class VLBIFrameBase(object):
    """Representation of a VLBI data frame, consisting of a header and payload.

    Parameters
    ----------
    header : VLBIHeaderBase
        Wrapper around the encoded header words, providing access to the
        header information.
    payload : VLBIPayloadBase
        Wrapper around the payload, provding mechanisms to decode it.
    valid : bool
        Whether this frame contains valid data (default: True).
    verify : bool
        Whether to do basic verification of integrity (default: True)

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
    to ``self.invalid_data_value``.

    A number of properties are defined: ``shape`` and ``dtype`` are the shape
    and type of the data array, ``words`` the full encoded frame, and ``size``
    the frame size in bytes.  Furthermore, the frame acts as a dictionary, with
    keys those of the header. Any attribute that is not defined on the frame
    itself, such as ``.time`` will be looked up on the header as well.
    """

    _header_class = None
    _payload_class = None
    invalid_data_value = 0.
    """Value used to replace data if the frame does not contain valid data."""

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
        assert (self.payloadsize ==
                self.payload.words.size * self.payload.words.dtype.itemsize)

    @property
    def valid(self):
        """Whether frame contains valid data."""
        return self._valid

    @valid.setter
    def valid(self, valid):
        self._valid = valid

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
        data : ndarray
            Array holding data to be encoded.
        header : VLBIHeaderBase
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
    def shape(self):
        """Shape of the data held in the payload (samples_per_frame, nchan)."""
        return self.payload.shape

    @property
    def dtype(self):
        """Numeric type of the payload."""
        return self.payload.dtype

    @property
    def size(self):
        """Size of the encoded frame in bytes."""
        return self.header.size + self.payload.size

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
        else:
            data = self.payload.__getitem__(item)
            if not self.valid:
                data[...] = self.invalid_data_value
            return data

    data = property(__getitem__,
                    doc="Decode the payload, zeroing it if not valid.")

    def __setitem__(self, item, value):
        if isinstance(item, six.string_types):
            self.header.__setitem__(item, value)
        else:
            self.payload.__setitem__(item, value)

    def keys(self):
        return self.header.keys()

    def __contains__(self, key):
        return key in self.keys()

    # Try to get any attribute not on the frame from the header properties.
    def __getattr__(self, attr):
        try:
            return self.__getattribute__(attr)
        except AttributeError:
            if attr in self.header._properties:
                return getattr(self.header, attr)
            else:
                raise

    # For tests, it is useful to define equality.
    def __eq__(self, other):
        return (type(self) is type(other) and
                self.valid == other.valid and
                self.header == other.header and
                self.payload == other.payload)
