# Licensed under the GPLv3 - see LICENSE
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
from astropy.extern import six

from ..vlbi_base.frame import VLBIFrameBase
from .header import VDIFHeader, VDIFBaseHeader
from .payload import VDIFPayload


__all__ = ['VDIFFrame', 'VDIFFrameSet']


class VDIFFrame(VLBIFrameBase):
    """Representation of a VDIF data frame, consisting of a header and payload.

    Parameters
    ----------
    header : `~baseband.vdif.VDIFHeader`
        Wrapper around the encoded header words, providing access to the
        header information.
    payload : `~baseband.vdif.VDIFPayload`
        Wrapper around the payload, provding mechanisms to decode it.
    valid : bool or None
        Whether the data are valid.  If `None` (default), is inferred from
        header.  Note that `header` is changed in-place if `True` or `False`.
    verify : bool
        Whether or not to do basic assertions that check the integrity
        (e.g., that channel information and whether or not data are complex
        are consistent between header and data).  Default: `True`.

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

    A number of properties are defined: `shape`, `dtype` and `size` are
    the shape, type and number of complete samples of the data array, and
    `nbytes` the frame size in bytes.  Furthermore, the frame acts as a
    dictionary, with keys those of the header.  Any attribute that is not
    defined on the frame itself, such as ``.time`` will be looked up on the
    header as well.
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
        edv : int, False, or None, optional
            Extended Data Version.  `False` is for legacy headers.  If `None`
            (default), it will be determined from the words themselves.
        verify : bool, optional
            Whether or not to do basic assertions that check the integrity
            (e.g., that channel information and whether or not data are complex
            are consistent between header and data).  Default: `True`.
        """
        header = cls._header_class.fromfile(fh, edv, verify)
        payload = cls._payload_class.fromfile(fh, header=header)
        # Since header was (optionally) verified, and payload was initialized
        # using header, frame verification is unnecessary.
        return cls(header, payload, verify=False)

    @classmethod
    def fromdata(cls, data, header=None, verify=True, **kwargs):
        """Construct frame from data and header.

        Parameters
        ----------
        data : `~numpy.ndarray`
            Array holding complex or real data to be encoded.
        header : `~baseband.vdif.VDIFHeader` or None
            If not given, will attempt to generate one using the keywords.
        verify : bool, optional
            Whether or not to do basic assertions that check the integrity
            (e.g., that channel information and whether or not data are complex
            are consistent between header and data). Default: `True`.
        **kwargs
            If ``header`` is not given, these are used to initialize one.
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
            m5h, nchan=m5pl.sample_shape.nchan, bps=m5pl.bps,
            invalid_data=not mark5b_frame.valid, **kwargs)
        payload = cls._payload_class(m5pl.words, header)
        return cls(header, payload, verify)


class VDIFFrameSet(object):
    """Representation of a set of VDIF frames, combining different threads.

    Parameters
    ----------
    frames : list of `~baseband.vdif.VDIFFrame`
        Should all cover the same time span.
    header0 : `~baseband.vdif.VDIFHeader`
        First header of the frame set.  If `None` (default), is extracted from
        ``frames[0]``.

    Notes
    -----
    The FrameSet can also be read instantiated using class methods:

      fromfile : read frames from a filehandle, optionally selecting threads

      fromdata : encode data as a set of frames

    Of course, one can also do the opposite:

      tofile : write frames to filehandle

      data : property that yields full decoded frame payloads

    One can decode part of the payload by indexing or slicing the frame.
    If the frame does not contain valid data, all values returned are set
    to ``self.fill_value``.

    A number of properties are defined: `shape`, `dtype` and `size` are
    the shape, type and number of complete samples of the data array, and
    `nbytes` the frame size in bytes.  Like a VDIFFrame, the frame set acts
    as a dictionary, with keys those of the header of the first frame
    (available via ``.header0``).  Any attribute that is not defined on the
    frame set itself, such as ``.time`` will also be looked up on the header.
    """
    def __init__(self, frames, header0=None):
        self.frames = frames
        if header0 is None:
            self.header0 = self.frames[0].header
        else:
            self.header0 = header0

    @classmethod
    def fromfile(cls, fh, thread_ids=None, edv=None, verify=True):
        """Read a frame set from a file, starting at the current location.

        Parameters
        ----------
        fh : filehandle
            Handle to the VDIF file.  Should be at the location where the
            frames are read from.
        thread_ids : list or None, optional
            The thread ids that should be read.  If `None` (default), continue
            reading threads as long as the frame number does not increase.
        edv : int or None, optional
            The expected extended data version for the VDIF Header.  If `None`
            (default), use that of the first frame.  (Passing it in slightly
            improves file integrity checking.)
        verify : bool, optional
            Whether to do (light) sanity checks on the header. Default: `True`.

        Returns
        -------
        frameset : `~baseband.vdif.VDIFFrameSet`
            Its ``frames`` property holds a list of frames (in order of either
            their ``thread_id`` or following the input ``thread_ids`` list).
            Use the `data` attribute to convert to an array.
        """
        header0 = VDIFHeader.fromfile(fh, edv, verify)
        edv = header0.edv
        frame_nr = header0['frame_nr']

        frames = {}
        header = header0
        while header['frame_nr'] == frame_nr:
            thread_id = header['thread_id']
            if thread_ids is None or thread_id in thread_ids:
                payload = VDIFPayload.fromfile(fh, header=header)
                frames[thread_id] = VDIFFrame(header, payload, verify=verify)
            else:
                fh.seek(header.payload_nbytes, 1)

            try:
                header = VDIFHeader.fromfile(fh, edv, verify)
            except EOFError:
                if thread_ids is None or len(frames) == len(thread_ids):
                    break
                else:
                    raise
        else:  # Move back to before header that had incorrect frame_nr.
            fh.seek(-header.nbytes, 1)

        if thread_ids and len(frames) < len(thread_ids):
            raise IOError("could not find all requested frames.")

        # Turn dict of frames into a list, following order given by
        # thread_ids, or just sorting by their own thread_id
        if thread_ids is None:
            thread_ids = sorted(frames.keys())
        frames = [frames[tid] for tid in thread_ids]

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
        data : `~numpy.ndarray`
            Array holding complex or real data to be encoded.  Dimensions
            should be (samples_per_frame, nthread, nchan).
        headers : `~baseband.vdif.VDIFHeader`, list of same, or None
            If a single header, a list with increasing ``thread_id`` is
            generated.  If not given, will attempt to generate a header from
            the keyword arguments.
        verify : bool
            Whether or not to do basic assertions that check the integrety
            (e.g., that channel information and whether or not data are complex
            are consistent between header and data).  Default: `True`.
        **kwargs
            If ``header`` is not given, these are used to initialize one.

        Returns
        -------
        frameset : `~baseband.vdif.VDIFFrameSet`
        """
        assert data.ndim == 3
        if not isinstance(headers, (list, tuple)):
            if headers is None:
                kwargs.setdefault('thread_id', 0)
                header = VDIFHeader.fromvalues(verify=verify, **kwargs)
            else:
                header = headers.copy()
            header['thread_id'] = 0
            headers = [header]
            for thread_id in range(1, data.shape[1]):
                header = header.copy()
                header['thread_id'] = thread_id
                headers.append(header)

        frames = [VDIFFrame.fromdata(d, h, verify)
                  for d, h in zip(data.transpose(1, 0, 2), headers)]
        return cls(frames)

    @property
    def nbytes(self):
        """Size of the encoded frame in bytes."""
        return len(self.frames) * self.frames[0].nbytes

    @property
    def sample_shape(self):
        """Shape of a sample in the frameset (nthread, nchan)."""
        return (len(self.frames),) + self.frames[0].sample_shape

    def __len__(self):
        """Number of samples in the frameset."""
        return len(self.frames[0])

    @property
    def shape(self):
        """Shape of the frameset data."""
        return (len(self),) + self.sample_shape

    @property
    def size(self):
        """Total number of component samples in the frameset data."""
        prod = 1
        for dim in self.shape:
            prod *= dim
        return prod

    @property
    def ndim(self):
        """Number of dimensions of the frameset data."""
        return len(self.shape)

    @property
    def dtype(self):
        """Numeric type of the frameset data."""
        return self.frames[0].dtype

    @property
    def valid(self):
        """Whether frameset contains valid data."""
        valid = np.array([frame.valid for frame in self.frames])
        return valid[0] if len(np.unique(valid)) == 1 else valid

    @valid.setter
    def valid(self, valid):
        valid = np.broadcast_to(valid, (len(self.frames),))
        for f, v in zip(self.frames, valid):
            f.valid = v

    @property
    def fill_value(self):
        """Value to replace invalid data in the frameset."""
        return self.frames[0].fill_value

    @fill_value.setter
    def fill_value(self, fill_value):
        for frame in self.frames:
            frame.fill_value = fill_value

    def _get_frames(self, item):
        """Get frames and other information required to obtain given item.

        Parameters
        ----------
        item : int, slice, or tuple
            Sample indices.  An int represents a single sample, a slice a
            sample range, and a tuple of ints/slices a range for multi-frame
            and multi-channel data.

        Returns
        -------
        frames : list
            List of frames needed for this slice.  For a list of length unity,
            ``single_frame`` determines whether it is an index or unit-length
            slice.
        frame_item : int, slice, or tuple
            The item that should be gotten/set for each frame
        single_sample, single_frame, single_channel : bool
            Whether the sample, frame, or channel axes are simple indices,
            and thus whether the corresponding dimension should be removed.

        Notes
        -----
        The sample part of ``item`` is restricted to (tuples of) ints or
        slices, so one cannot access non-contiguous samples using advanced
        indexing.  Futhermore, if ``item`` is a slice, a negative increment
        cannot be used.  The function is unable to parse payloads whose words
        have unused space (eg. VDIF files with 20 bits/sample).
        """
        if item is ():
            return self.frames, (), False, False, False

        if not isinstance(item, tuple):
            return self.frames, item, not isinstance(item, slice), False, False

        single_sample = not isinstance(item[0], slice)
        if len(item) == 1:
            return self.frames, False, item[0], single_sample, False

        single_channel = (len(item) > 2 and
                          np.empty(self.shape[2:])[item[2:]].ndim == 0)
        frame_indices = np.arange(len(self.frames))[item[1]]
        assert frame_indices.ndim <= 1
        single_frame = frame_indices.ndim == 0
        frames = [self.frames[i] for i in np.atleast_1d(frame_indices)]
        return (frames, item[:1] + item[2:],
                single_sample, single_frame, single_channel)

    # Header behaves as a dictionary, while Payload can be indexed/sliced.
    # Let frameset behave appropriately.
    def __getitem__(self, item=()):
        if isinstance(item, six.string_types):
            # Header behaves as a dictionary.  Assume base keywords are the
            # same for every frame, except for thread_id (must be different)
            # and invalid_data (can be different).
            if item == 'thread_id':
                return np.array([f.header[item] for f in self.frames])
            elif (item != 'invalid_data' and
                  item in VDIFBaseHeader._header_parser.keys()):
                return self.header0[item]

            values = np.array([f.header[item] for f in self.frames])
            return values[0] if len(np.unique(values)) == 1 else values

        (frames, frame_item,
         single_sample, single_frame, single_channel) = self._get_frames(item)

        data0 = frames[0][frame_item]
        if single_frame:
            return data0

        if single_sample:
            swapped = data = np.empty((len(frames),) + data0.shape,
                                      dtype=self.dtype)
        else:
            data = np.empty((data0.shape[0], len(frames)) +
                            data0.shape[1:], dtype=self.dtype)
            swapped = data.swapaxes(0, 1)

        swapped[0] = data0
        for frame, frame_data in zip(frames[1:], swapped[1:]):
            frame_data[...] = frame[frame_item]
        return data

    def __setitem__(self, item, data):
        if isinstance(item, six.string_types):
            # Headers behave as dictionaries; except for thread_id and
            # invalid_data, assume set base properties all to the same value.
            data = np.broadcast_to(data, (len(self.frames),))
            if item == 'thread_id':
                if len(np.unique(data)) != len(self.frames):
                    raise ValueError("all thread ids should be unique.")
            elif (item != 'invalid_data' and
                  item in VDIFBaseHeader._header_parser.keys()):
                if data.strides != (0,) and len(np.unique(data)) > 1:
                    raise ValueError("base header keys should be identical.")

            for f, value in zip(self.frames, data):
                f.header[item] = value
            return

        (frames, frame_item,
         single_sample, single_frame, single_channel) = self._get_frames(item)

        if single_frame:
            frames[0][frame_item] = data
            return

        data = np.asanyarray(data)
        if single_channel:
            if single_sample or data.ndim <= 1:
                swapped = np.broadcast_to(data, (len(frames),))
            else:
                new_shape = (data.shape[0], len(frames))
                swapped = np.broadcast_to(data, new_shape).swapaxes(0, 1)
        else:
            if single_sample or data.ndim <= 2:
                new_shape = (len(frames),) + data.shape[1:]
                swapped = np.broadcast_to(data, new_shape)
            else:
                new_shape = (data.shape[0], len(frames)) + data.shape[2:]
                swapped = np.broadcast_to(data, new_shape).swapaxes(0, 1)

        for frame, frame_data in zip(frames, swapped):
            frame[frame_item] = frame_data

    data = property(__getitem__, doc="Full decoded frame.")

    def keys(self):
        return self.header0.keys()

    def _ipython_key_completions_(self):
        # Enables tab-completion of header keys in IPython.
        return self.keys()

    def __contains__(self, key):
        return key in self.header0

    def __getattr__(self, attr):
        if attr in self.header0._properties:
            if attr in VDIFBaseHeader._properties:
                return getattr(self.header0, attr)
            values = np.array([getattr(f.header, attr) for f in self.frames])
            return values[0] if len(np.unique(values)) == 1 else values
        else:
            return self.__getattribute__(attr)

    # For tests, it is useful to define equality.
    def __eq__(self, other):
        return (type(self) is type(other) and
                len(self.frames) == len(other.frames) and
                self.header0 == other.header0 and
                all(f1 == f2 for f1, f2 in zip(self.frames, other.frames)))
