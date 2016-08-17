"""
Definitions for VLBI VDIF frames and frame sets.

Implements a VDIFFrame class  that can be used to hold a header and a
payload, providing access to the values encoded in both.  Also, define
a VDIFFrameSet class that combines a set of frames from different threads.

For the VDIF specification, see http://www.vlbi.org/vdif
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from ..vlbi_base.frame import VLBIFrameBase
from .header import VDIFHeader
from .payload import VDIFPayload


__all__ = ['VDIFFrame', 'VDIFFrameSet']


class VDIFFrame(VLBIFrameBase):
    """Representation of a VDIF data frame, consisting of a header and payload.

    Parameters
    ----------
    header : VDIFHeader
        Wrapper around the encoded header words, providing access to the
        header information.
    payload : VDIFPayload
        Wrapper around the payload, provding mechanisms to decode it.
    valid : bool or `None`
        Whether the data is valid.  If `None` (default), inferred from header.
        Note that header is changed in-place if `True` or `False`.
    verify : bool
        Whether or not to do basic assertions that check the integrity
        (e.g., that channel information and whether or not data are complex
        are consistent between header and data).  Default: `True`

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
    and type of the data array, and ``size`` the frame size in bytes.
    Furthermore, the frame acts as a dictionary, with keys those of the header.
    Any attribute that is not defined on the frame itself, such as ``.time``
    will be looked up on the header as well.
    """

    _header_class = VDIFHeader
    _payload_class = VDIFPayload

    def __init__(self, header, payload, valid=None, verify=True):
        self.header = header
        self.payload = payload
        if valid is not None:
            self.valid = valid
        if verify:
            self.verify()

    def verify(self):
        """Verify integrity.

        Checks consistency between the header information and payload
        data shape and type.
        """
        super(VDIFFrame, self).verify()
        assert self.header['complex_data'] == (self.payload.dtype.kind == 'c')
        assert self.payload.shape == (self.header.samples_per_frame,
                                      self.header.nchan)

    @property
    def valid(self):
        """Whether frame contains valid data.

        This is just the opposite of the ``invalid_data`` item in the header.
        If set, that header item is adjusted correspondingly.
        """
        return not self.header['invalid_data']

    @valid.setter
    def valid(self, valid):
        self.header['invalid_data'] = not valid

    @classmethod
    def fromfile(cls, fh, edv=None, verify=True):
        """Read a frame from a filehandle.

        Parameters
        ----------
        fh : filehandle
            From which the header and payload are read.
        edv : int, False, or None.
            VDIF Extended Data Version.  ``False`` is for legacy headers.
            If ``None``, it will be determined from the words itself.
        verify : bool
            Whether or not to do basic assertions that check the integrity
            (e.g., that channel information and whether or not data are complex
            are consistent between header and data).  Default: `True`.
        """
        header = cls._header_class.fromfile(fh, edv, verify)
        payload = cls._payload_class.fromfile(fh, header=header)
        return cls(header, payload, verify=verify)

    @classmethod
    def fromdata(cls, data, header=None, verify=True, **kwargs):
        """Construct frame from data and header.

        Parameters
        ----------
        data : ndarray
            Array holding complex or real data to be encoded.
        header : VDIFHeader or None
            If `None`, it will be attemtped to create one using the keywords.
        verify : bool
            Whether or not to do basic assertions that check the integrity
            (e.g., that channel information and whether or not data are complex
            are consistent between header and data). Default: `True`.
        """
        if header is None:
            header = cls._header_class.fromvalues(verify=verify, **kwargs)

        payload = cls._payload_class.fromdata(data, header=header)

        return cls(header, payload, verify=True)

    @classmethod
    def from_mark5b_frame(cls, mark5b_frame, verify=True, **kwargs):
        """Construct an Mark5B over VDIF frame (EDV=0xab).

        Any additional keywords can be used to set VDIF header properties
        not found in the Mark 5B header (such as station).

        See http://www.vlbi.org/vdif/docs/vdif_extension_0xab.pdf
        """
        m5h, m5pl = mark5b_frame.header, mark5b_frame.payload
        header = cls._header_class.from_mark5b_header(
            m5h, nchan=m5pl.nchan, bps=m5pl.bps,
            invalid_data=not mark5b_frame.valid, **kwargs)
        payload = cls._payload_class(m5pl.words, header)
        return cls(header, payload, verify)


class VDIFFrameSet(object):
    """Representation of a set of VDIF frames, combining different threads.

    Parameters
    ----------
    frames : list of VDIFFrame instances
        Should all cover the same time span.

    The FrameSet can also be read instantiated using class methods:

      fromfile : read frames from a filehandle, optionally selecting threads.

      fromdata : encode data as a set of frames

    Of course, one can also do the opposite:

      tofile : write frames to filehandle

      data : property that yields full decoded frame payloads

    One can decode part of the payload by indexing or slicing the frame.
    If the frame does not contain valid data, all values returned are set
    to ``self.invalid_data_value``.

    A number of properties are defined: ``shape`` and ``dtype`` are the shape
    and type of the data array, and ``size`` the total size in bytes.  Like a
    VDIFFrame, the frame set acts as a dictionary, with keys those of the
    header of the first frame (available via ``.header0``).  Any attribute that
    is not defined on the frame set itself, such as ``.time`` will also be
    looked up on the header.
    """
    invalid_data_value = 0.

    def __init__(self, frames, header0=None):
        self.frames = frames
        # Used in .data below to decode data only once.
        self._data = None
        if header0 is None:
            self.header0 = self.frames[0].header
        else:
            self.header0 = header0

    @classmethod
    def fromfile(cls, fh, thread_ids=None, sort=True, edv=None, verify=True):
        """Read a frame set from a file, starting at the current location.

        Parameters
        ----------
        fh : filehandle
            Handle to the VDIF file.  Should be at the location where the
            frames are read from.
        thread_ids : list or None
            The thread ids that should be read.  If `None`, continue reading
            threads as long as the frame number does not increase.
        sort : bool
            Whether to sort the frames by thread_id.  Default: True.
            Note that this does not influence the header used to look up
            attributes (it is always the header of the first frame read).
            It does, however, influence the order in which decoded data is
            returned.
        edv : int or None
            The expected extended data version for the VDIF Header.  If not
            given, use that of the first frame.  (Passing it in slightly
            improves file integrity checking.)
        verify : bool
            Whether to do (light) sanity checks on the header. Default: True.

        Returns
        -------
        frameset : VDIFFrameSet instance
            Holds ''frames'' property with a possibly sorted list of frames.
            Use the ''data'' attribute to convert to an array.
        """
        header0 = VDIFHeader.fromfile(fh, edv, verify)
        edv = header0.edv

        frames = []
        header = header0
        while header['frame_nr'] == header0['frame_nr']:
            if thread_ids is None or header['thread_id'] in thread_ids:
                frames.append(
                    VDIFFrame(header, VDIFPayload.fromfile(fh, header=header),
                              verify=verify))
            else:
                fh.seek(header.payloadsize, 1)

            try:
                header = VDIFHeader.fromfile(fh, edv, verify)
            except EOFError:
                if thread_ids is None or len(frames) == len(thread_ids):
                    break
                else:
                    raise
        else:  # Move back to before header that had incorrect frame_nr.
            fh.seek(-header.size, 1)

        if thread_ids is None:
            thread_ids = range(min(len(frames), 1))

        if len(frames) < len(thread_ids):
            raise IOError("Could not find all requested frames.")

        if sort:
            frames.sort(key=lambda frame: frame['thread_id'])

        return cls(frames, header0)

    def tofile(self, fh):
        """Write all encoded frames to filehandle."""
        for frame in self.frames:
            frame.tofile(fh)

    @classmethod
    def fromdata(cls, data, headers=None, verify=True, **kwargs):
        """Construct a set of frames from data and headers.

        Parameters
        ----------
        data : ndarray
            Array holding complex or real data to be encoded.  Dimensions
            should be (nthread, nsample, nchan).
        headers : list of VDIFHeader instances, VDIFHeader or None
            If a single header, a list with increasing ``thread_id`` is
            generated. If `None`, it is attempted to generate a header from
            the keyword arguments.
        verify : bool
            Whether or not to do basic assertions that check the integrety
            (e.g., that channel information and whether or not data are complex
            are consistent between header and data).

        Returns
        -------
        frameset : VDIFFrameSet instance.
        """
        if not isinstance(headers, (list, tuple)):
            if headers is None:
                kwargs.setdefault('thread_id', 0)
                header = VDIFHeader.fromvalues(verify=verify, **kwargs)
            else:
                header = headers.copy()
            header['thread_id'] = 0
            headers = [header]
            for thread_id in range(1, len(data)):
                header = header.copy()
                header['thread_id'] = thread_id
                headers.append(header)

        frames = [VDIFFrame.fromdata(d, h, verify)
                  for d, h in zip(data, headers)]
        return cls(frames)

    @property
    def data(self):
        """Decode the payload."""
        if self._data is None:
            self._data = np.empty(self.shape, dtype=self.dtype)
            for frame, datum in zip(self.frames, self._data):
                datum[...] = (frame.data if frame.valid else
                              self.invalid_data_value)
        return self._data

    @property
    def size(self):
        return len(self.frames) * self.frames[0].size

    @property
    def shape(self):
        return (len(self.frames),) + self.frames[0].shape

    @property
    def dtype(self):
        return self.frames[0].dtype

    def __getitem__(self, item):
        # Header behaves as a dictionary.
        return self.header0.__getitem__(item)

    def keys(self):
        return self.header0.keys()

    def __contains__(self, key):
        return key in self.header0

    def __getattr__(self, attr):
        try:
            return self.__getattribute__(attr)
        except AttributeError:
            if attr in self.header0._properties:
                return getattr(self.header0, attr)
            else:
                raise

    # For tests, it is useful to define equality.
    def __eq__(self, other):
        return (type(self) is type(other) and
                len(self.frames) == len(other.frames) and
                self.header0 == other.header0 and
                all(f1 == f2 for f1, f2 in zip(self.frames, other.frames)))
