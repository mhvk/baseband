from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from ..vlbi_base import VLBIFrameBase
from .header import VDIFHeader
from .payload import VDIFPayload


class VDIFFrame(VLBIFrameBase):
    """Representation of a VDIF data frame, consisting of a header and payload.

    Parameters
    ----------
    header : VDIFHeader
        Wrapper around the encoded header words, providing access to the
        header information.
    payload : VDIFPayload
        Wrapper around the payload, provding mechanisms to decode it.
    verify : bool
        Whether or not to do basic assertions that check the integrity
        (e.g., that channel information and whether or not data are complex
        are consistent between header and data).  Default: `True`

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

    _header_class = VDIFHeader
    _payload_class = VDIFPayload

    def __init__(self, header, payload, verify=True):
        self.header = header
        self.payload = payload
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
        header = VDIFHeader.fromfile(fh, edv, verify)
        payload = VDIFPayload.fromfile(fh, header=header)
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
            header = VDIFHeader.fromvalues(verify=verify, **kwargs)

        payload = VDIFPayload.fromdata(data, header=header)

        return cls(header, payload, verify=True)

    @classmethod
    def from_mark5b_frame(cls, mark5b_frame, verify=True):
        """Construct an Mark5B over VDIF frame (EDV=0xab).

        See http://www.vlbi.org/vdif/docs/vdif_extension_0xab.pdf
        """
        m5h, m5pl = mark5b_frame.header, mark5b_frame.payload
        header = VDIFHeader.from_mark5b_header(
            m5h, nchan=m5pl.nchan, bps=m5pl.bps,
            invalid_data=not mark5b_frame.valid)
        payload = VDIFPayload(m5pl.words, header)
        return cls(header, payload, verify)


class VDIFFrameSet(object):
    def __init__(self, frames, header0=None):
        self.frames = frames
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
            The thread ids that should be read.  If `None`, read all threads.
        sort : bool
            Whether to sort the frames by thread_id.  Default: True.
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
        exc = None
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
            except EOFError as exc:
                fh.seek(0, 2)
                break
        else:  # Move back to before header that had incorrect frame_nr.
            fh.seek(-header.size, 1)

        if thread_ids is None:
            thread_ids = range(min(len(frames), 1))

        if len(frames) < len(thread_ids):
            if exc is not None:
                raise
            else:
                raise IOError("Could not find all requested frames.")

        if sort:
            frames.sort(key=lambda frame: frame['thread_id'])

        return cls(frames, header0)

    def tofile(self, fh):
        for frame in self.frames:
            frame.tofile(fh)

    @classmethod
    def fromdata(cls, data, headers, verify=True):
        """Construct a set of frames from data and headers.

        Parameters
        ----------
        data : ndarray
            Array holding complex or real data to be encoded.  Dimensions
            should be (nthread, nsample, nchan).
        headers : list of VDIFHeader instances, VDIFHeader or dict
            If a single header (or dict with relevant keywords), a list with
            increasing ``thread_id`` is generated.
        verify : bool
            Whether or not to do basic assertions that check the integrety
            (e.g., that channel information and whether or not data are complex
            are consistent between header and data).

        Returns
        -------
        frameset : VDIFFrameSet instance.
        """
        if not isinstance(headers, (list, tuple)):
            header = (headers if isinstance(headers, VDIFHeader)
                      else VDIFHeader.fromvalues(**headers))
            headers = []
            for thread_id in range(len(data)):
                header = header.copy()
                header['thread_id'] = thread_id
                headers.append(header)

        frames = [VDIFFrame.fromdata(d, h, verify)
                  for d, h in zip(data, headers)]
        return cls(frames)

    def todata(self, data=None, invalid_data_value=0.):
        """Decode the payload.

        Parameters
        data : None or ndarray
            If given, the data is decoded into the array (which should have
            the correct shape).  By default, a new array is created, which is
            kept for other invocations (i.e., decoding is only done once).
        invalid_data_value : float
            Value to use for invalid data frames (default: 0.).
        """
        if data is None:
            if self._data is not None:
                return self._data

            data = np.empty(self.shape, dtype=self.dtype)

        for frame, datum in zip(self.frames, data):
            frame.todata(datum, invalid_data_value)

        self._data = data
        return data

    data = property(todata, doc="Decode the payloads in all frames.")

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
        return key in self.header[0].keys()

    def __getattr__(self, attr):
        try:
            return self.__getattribute__(attr)
        except AttributeError:
            if attr in self.header0._properties:
                return getattr(self.header0, attr)
            else:
                raise
