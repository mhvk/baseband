from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import io
import numpy as np

from .header import VDIFHeader
from .payload import VDIFPayload


class VDIFFrame(object):
    def __init__(self, header, payload, verify=True):
        self.header = header
        self.payload = payload
        if verify:
            self.verify()

    def verify(self):
        assert self.payloadsize // 4 == self.payload.words.size

    @classmethod
    def frombytes(cls, raw, edv=None, verify=True):
        # Assume non-legacy header to ensure those are done fastest.
        header = VDIFHeader.frombytes(raw[:32], edv, verify=False)
        if header.edv is False:
            header.words = header.words[:4]
            payload = raw[16:]
        else:
            payload = raw[32:]
        if verify:
            header.verify()

        payload = VDIFPayload.frombytes(payload, header=header)
        return cls(header, payload, verify)

    def tobytes(self):
        return self.header.tobytes() + self.payload.tobytes()

    @classmethod
    def fromfile(cls, fh, edv=None, verify=True):
        header = VDIFHeader.fromfile(fh, edv, verify)
        payload = VDIFPayload.fromfile(fh, header=header)
        return cls(header, payload, verify)

    def tofile(self, fh):
        return fh.write(self.tobytes())

    @classmethod
    def fromdata(cls, data, header, verify=True):
        """Construct frame from data and header.

        Parameters
        ----------
        data : ndarray
            Array holding complex or real data to be encoded.
        header : VDIFHeader or dict
            Header or dict with relevant keywords to construct one.
        verify : bool
            Whether or not to do basic assertions that check the integrety
            (e.g., that channel information and whether or not data are complex
            are consistent between header and data).

        Returns
        -------
        frame : VDIFFrame instance.
        """
        if not isinstance(header, VDIFHeader):
            header = VDIFHeader.fromvalues(**header)

        if verify:
            assert header['complex_data'] == (data.dtype.kind == 'c')
            assert header.nchan == data.shape[-1]

        payload = VDIFPayload.fromdata(data, header=header)

        return cls(header, payload, verify)

    def todata(self, data=None):
        return self.payload.todata(data)

    data = property(todata, doc="Decode the payload")

    @property
    def shape(self):
        return self.payload.shape

    @property
    def dtype(self):
        return self.payload.dtype

    @property
    def words(self):
        return np.hstack((np.array(self.header.words), self.payload.words))

    @property
    def size(self):
        return self.header.size + self.payload.size

    def __array__(self):
        return self.payload.data

    def __getitem__(self, item):
        # Header behaves as a dictionary.
        return self.header.__getitem__(item)

    def keys(self):
        return self.header.keys()

    def __contains__(self, key):
        return key in self.header.keys()

    def __getattr__(self, attr):
        try:
            return self.__getattribute__(attr)
        except AttributeError:
            if attr in self.header._properties:
                return getattr(self.header, attr)


class VDIFFrameSet(object):
    def __init__(self, frames, header0=None):
        self.frames = frames
        self._data = None
        if header0 is None:
            self.header0 = self.frames[0].header
        else:
            self.header0 = header0

    @classmethod
    def frombytes(cls, raw, *args, **kwargs):
        """Read a frame set from a byte string.

        Implemented via ``fromfile`` using BytesIO.  For reading from files,
        use ``fromfile`` directly.
        """
        return cls.fromfile(io.BytesIO(raw), *args, **kwargs)

    def tobytes(self):
        return b''.join(frame.tobytes() for frame in self.frames)

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

    def todata(self, data=None):
        if data is None:
            if self._data is not None:
                return self._data

            data = np.empty(self.shape, dtype=self.dtype)

        for frame, datum in zip(self.frames, data):
            frame.todata(datum)

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
