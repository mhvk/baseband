"""
Base definitions for VLBI frames, used for VDIF and Mark 5B.

Defines a frame class VLBIFrameBase that can be used to hold a header and a
payload, providing access to the values encoded in both.
"""
# Helper functions for VLBI readers (VDIF, Mark5B).
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np


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

    The Frame can also be read instantiated using class methods:

      fromfile : read header and payload from a filehandle

      fromdata : encode data as payload

    It also has methods to do the opposite:

      tofile : write header and payload to filehandle

      todata : decode payload to data

    A number of properties are defined: ``shape`` and ``dtype`` are the shape
    and type of the data array, ``words`` the full encoded frame, and ``size``
    the frame size in bytes.  Furthermore, the frame acts as a dictionary, with
    keys those of the header. Any attribute that is not defined on the frame
    itself, such as ``.time`` will be looked up on the header as well.
    """

    _header_class = None
    _payload_class = None

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
        assert self.payloadsize // 4 == self.payload.words.size

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

        Any arguments beyond the filehandle are used to help initialize the
        payload, except for ``valid`` and ``verify``, which are passed on to
        the header and class initializers.
        """
        valid = kwargs.pop('valid', True)
        verify = kwargs.pop('verify', True)
        payload = cls._payload_class.fromdata(data, *args, **kwargs)
        return cls(header, payload, valid=valid, verify=verify)

    def todata(self, data=None, invalid_data_value=0.):
        """Decode the payload.

        Parameters
        data : None or ndarray
            If given, the data is decoded into the array (which should have
            the correct shape).  By default, a new array is created.
        invalid_data_value : float
            Value to use for invalid data frames (default: 0.).
        """
        out = self.payload.todata(data)
        if not self.valid:
            out[...] = self._invalid_data_value
        return out

    data = property(todata, doc="Decode the payload, zeroing it if not valid.")

    @property
    def shape(self):
        """Shape of the data held in the payload (samples_per_frame, nchan)."""
        return self.payload.shape

    @property
    def dtype(self):
        """Numeric type of the payload."""
        return self.payload.dtype

    @property
    def words(self):
        """Frame encoded in unsigned 32-bit integers."""
        return np.hstack((np.array(self.header.words), self.payload.words))

    @property
    def size(self):
        """Size of the encoded frame in bytes."""
        return self.header.size + self.payload.size

    def __array__(self):
        """Interface to arrays."""
        return self.data

    # Header behaves as a dictionary.  Let frame behave the same.
    def __getitem__(self, item):
        return self.header.__getitem__(item)

    def keys(self):
        return self.header.keys()

    def __contains__(self, key):
        return key in self.header.keys()

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
