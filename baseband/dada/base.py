# Licensed under the GPLv3 - see LICENSE.rst
import io

import numpy as np

from ..vlbi_base.base import (VLBIStreamBase, VLBIStreamReaderBase,
                              VLBIStreamWriterBase)
from .header import DADAHeader
from .payload import DADAPayload
from .frame import DADAFrame


__all__ = ['DADAFileReader', 'DADAFileWriter',
           'DADAStreamBase', 'DADAStreamReader', 'DADAStreamWriter', 'open']


class DADAFileReader(io.BufferedReader):
    """Simple reader for DADA files.

    Adds a ``read_frame`` method to the basic binary file reader
    :class:`~io.BufferedReader`.  By default, the payload is mapped
    rather than fully read into physical memory.
    """
    def read_frame(self, memmap=True):
        """Read the frame header and read or map the corresponding payload.

        Parameters
        ----------
        memmap : bool, optional
            Whether to map the payload on disk using `~numpy.memmap`, so that
            parts are only loaded into memory as needed to access data.

        Returns
        -------
        frame : `~baseband.dada.DADAFrame`
            With a ``.header`` and ``.payload`` properties.  The ``.data``
            property returns all data encoded in the frame.  Since this may
            be too large to fit in memory, it may be better to access the
            parts of interest by slicing the frame.
        """
        return DADAFrame.fromfile(self, memmap=memmap)


class DADAFileWriter(io.BufferedRandom):
    """Simple writer/mapper for DADA files.

    Adds ``write_frame`` and ``memmap_frame`` methods to the basic file
    reader/writer :class:`~io.BufferedRandom`.  The latter allows one to
    encode data in pieces, writing to disk as needed.
    """
    def write_frame(self, data, header=None, **kwargs):
        """Write a single frame (header plus payload).

        Parameters
        ----------
        data : array or `~baseband.dada.DADAFrame`
            If an array, a ``header`` should be given, which will be used to
            get the information needed to encode the array, and to construct
            the DADA frame.
        header : `~baseband.dada.DADAHeader`, optional
            Ignored if `data` is a DADA Frame.
        **kwargs
            If no `header` is given, an attempt is made to initialize one
            using keywords arguments.
        """
        if not isinstance(data, DADAFrame):
            data = DADAFrame.fromdata(data, header, **kwargs)
        return data.tofile(self)

    def memmap_frame(self, header=None, **kwargs):
        if header is None:
            header = DADAHeader.fromvalues(**kwargs)
        header.tofile(self)
        payload = DADAPayload.fromfile(self, memmap=True, header=header)
        return DADAFrame(header, payload)


class DADAStreamBase(VLBIStreamBase):
    """DADA file wrapper, which combines threads into streams."""

    def __init__(self, fh_raw, header0, thread_ids=None):
        frames_per_second = (1. / (header0['TSAMP'] * 1e-6) /
                             header0.samples_per_frame)
        if thread_ids is None:
            thread_ids = list(range(header0['NPOL']))
        super(DADAStreamBase, self).__init__(
            fh_raw=fh_raw, header0=header0, nchan=header0['NCHAN'],
            bps=header0.bps, thread_ids=thread_ids,
            samples_per_frame=header0.samples_per_frame,
            frames_per_second=frames_per_second)
        self._frame_nr = 0

    def _frame_info(self):
        """Convert offset to file number and offset into that file."""
        return divmod(self.offset, self.samples_per_frame)


class DADAStreamReader(DADAStreamBase, VLBIStreamReaderBase):
    """DADA format reader.

    This wrapper allows one to access a DADA file as a continues series of
    samples.

    Parameters
    ----------
    raw : filehandle
        file handle of the raw DADA stream
    thread_ids: list of int, optional
        Specific threads to read.  By default, all threads are read.
    """
    def __init__(self, raw, thread_ids=None):
        self._frame = raw.read_frame(memmap=True)
        header = self._frame.header
        super(DADAStreamReader, self).__init__(raw, header, thread_ids)

    @property
    def header1(self):
        """Last header of the file."""
        return self.header0

    def read(self, count=None, squeeze=True, out=None):
        """Read count samples.

        Parameters
        ----------
        count : int
            Number of samples to read.  If omitted or negative, the whole
            file is read.
        squeeze : bool
            If `True` (default), remove channel and thread dimensions if unity.
        out : `None` or array
            Array to store the data in. If given, count will be inferred,
            and squeeze is set to `False`.

        Returns
        -------
        out : array of float or complex
            Dimensions are (sample-time, thread, channel).
        """
        if out is None:
            if count is None or count < 0:
                count = self.size - self.offset

            out = np.empty((count,) + self._frame.sample_shape,
                           dtype=self._frame.dtype)
        else:
            count = out.shape[0]
            squeeze = False

        offset0 = self.offset
        while count > 0:
            frame_nr, sample_offset = self._frame_info()
            if frame_nr != self._frame_nr:
                # Open relevant file.
                self._read_frame()
                assert (self._frame.header['OBS_OFFSET'] ==
                        self.header0['OBS_OFFSET'] + frame_nr *
                        self.header0.payloadsize)

            # Copy relevant data from frame into output.
            nsample = min(count, self.samples_per_frame - sample_offset)
            sample = self.offset - offset0
            data_slice = slice(sample_offset, sample_offset + nsample)
            if self.thread_ids:
                data_slice = (data_slice, self.thread_ids)

            out[sample:sample + nsample] = self._frame[data_slice]
            self.offset += nsample
            count -= nsample

        return out.squeeze() if squeeze else out

    def _read_frame(self):
        frame_nr = self.offset // self.samples_per_frame
        if frame_nr != self._frame_nr:
            self.fh_raw.close()
            self.fh_raw = DADAFileReader(io.open(self.files[frame_nr], 'rb'))
        self._frame = self.fh_raw.read_frame(memmap=True)
        self._frame_nr = frame_nr


class DADAStreamWriter(DADAStreamBase, VLBIStreamWriterBase):
    """DADA format writer.

    Parameters
    ----------
    raw : filehandle
        For writing the header and raw data.
    header : :class:`~baseband.dada.DADAHeader`, optional
        Header for the file, holding time information, etc.
    **kwargs
        If no header is give, an attempt is made to construct the header from
        these.  For a standard header, this would include the following.

    --- Header keywords : (see :meth:`~baseband.dada.DADAHeader.fromvalues`)

    time : `~astropy.time.Time`
        The start time of this file.
    offset : `~astropy.units.Quantity`
        A possible offset from the start of the whole observation.
    nchan : int, optional
        Number of FFT channels within stream (default 1).
        Note: that different # of channels per thread is not supported.
    complex_data : bool
        Whether data is complex
    bps : int
        Bits per sample (or real, imaginary component).
    samples_per_frame : int
        Number of complete samples in a given frame.
    """
    def __init__(self, raw, header=None, **kwargs):
        if header is None:
            header = DADAHeader.fromvalues(**kwargs)
        super(DADAStreamWriter, self).__init__(raw, header)
        self.header0.tofile(self.fh_raw)

    def write(self, data, squeezed=True, invalid_data=False):
        """Write data, buffering by frames as needed."""
        if squeezed and data.ndim < 3:
            if self.nthread == 1:
                data = np.expand_dims(data, axis=1)
            if self.nchan == 1:
                data = np.expand_dims(data, axis=-1)

        assert data.shape[1] == self.nthread
        assert data.shape[2] == self.nchan

        payload = DADAPayload.fromdata(data, bps=self.header0.bps)
        payload.tofile(self.fh_raw)
        self.offset += data.shape[0]


def open(name, mode='rs', *args, **kwargs):
    """Open DADA format file for reading or writing.

    Opened as binary, one gets a standard file handle.  Opened as a stream, the
    file handle is wrapped, allowing access to it as a series of samples.

    Parameters
    ----------
    name : str
        File name
    mode : {'rb', 'wb', 'rs', or 'ws'}, optional
        Whether to open for reading or writing, and as a regular binary file
        or as a stream (default is reading a stream).
    **kwargs :
        Additional arguments when opening the file as a stream

    --- For reading : (see :class:`~baseband.dada.DADAStreamReader`)

    thread_ids : list of int, optional
        Specific threads to read.  By default, all threads are read.

    --- For writing : (see :class:`~baseband.dada.DADAStreamWriter`)

    header : `~baseband.dada.DADAHeader`, optional
        Header for the first frame, holding time information, etc.
    **kwargs
        If the header is not given, an attempt will be made to construct one
        with any further keyword arguments.  See
        :class:`~baseband.dada.DADAStreamWriter`.

    Returns
    -------
    Filehandle
        :class:`~baseband.dada.base.DADAFileReader` or
        :class:`~baseband.dada.base.DADAFileWriter` instance (binary), or
        :class:`~baseband.dada.base.DADAStreamReader` or
        :class:`~baseband.dada.base.DADAStreamWriter` instance (stream).
    """
    # Typical name 2016-04-23-07:29:30_0000000000000000.000000.dada
    if 'w' in mode:
        if not hasattr(name, 'write'):
            name = io.open(name, 'w+b')
        fh = DADAFileWriter(name)
        return fh if 'b' in mode else DADAStreamWriter(fh, **kwargs)
    elif 'r' in mode:
        if not hasattr(name, 'read'):
            name = io.open(name, 'rb')
        fh = DADAFileReader(name)
        return fh if 'b' in mode else DADAStreamReader(fh, **kwargs)
    else:
        raise ValueError("Only support opening DADA file for reading "
                         "or writing (mode='r' or 'w').")
