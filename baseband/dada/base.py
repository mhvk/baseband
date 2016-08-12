# Licensed under the GPLv3 - see LICENSE.rst
import io

import numpy as np

from ..vlbi_base.base import (VLBIStreamBase, VLBIStreamReaderBase,
                              VLBIStreamWriterBase)
from .header import DADAHeader
from .payload import DADAPayload


__all__ = ['DADAStreamBase', 'DADAStreamReader', 'DADAStreamWriter', 'open']


class DADAStreamBase(VLBIStreamBase):
    """DADA file wrapper, which combines threads into streams."""

    def __init__(self, fh_raw, header0, thread_ids=None):
        frames_per_second = (1. / (header0['TSAMP'] * 1e-6) /
                             header0.samples_per_frame)
        if thread_ids is None:
            thread_ids = [range(header0['NPOL'])]
        super(DADAStreamBase, self).__init__(
            fh_raw=fh_raw, header0=header0, nchan=header0['NCHAN'],
            bps=header0.bps, thread_ids=thread_ids,
            samples_per_frame=header0.samples_per_frame,
            frames_per_second=frames_per_second)


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
        header = DADAHeader.fromfile(raw)
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

        else:
            count = out.shape[0]
            squeeze = False

        bytes_per_sample = self.header0.payloadsize // self.size
        # Don't know if this can every not be true.  Too lazy to check.
        assert bytes_per_sample * self.size == self.header0.payloadsize
        self.fh_raw.seek(self.offset * bytes_per_sample + self.header0.size)
        payload = DADAPayload.fromfile(self.fh_raw,
                                       payloadsize=count * bytes_per_sample,
                                       bps=self.header0.bps,
                                       complex_data=self.header0.complex_data,
                                       sample_shape=(self.header0['NPOL'],
                                                     self.header0['NCHAN']))

        out = payload.todata(data=out)
        self.offset += count
        return out.squeeze() if squeeze else out


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

    --- For reading : (see :class:`DADAStreamReader`)

    thread_ids : list of int, optional
        Specific threads to read.  By default, all threads are read.

    --- For writing : (see :class:`DADAStreamWriter`)

    header : `~baseband.dada.DADAHeader`, optional
        Header for the first frame, holding time information, etc.
    **kwargs
        If the header is not given, an attempt will be made to construct one
        with any further keyword arguments.  See :class:`DADAStreamWriter`.

    Returns
    -------
    Filehandle
        A regular filehandle (binary), or a :class:`DADAStreamReader` or
        :class:`DADAStreamWriter` instance (stream).
    """
    if 'w' in mode:
        if not hasattr(name, 'write'):
            name = io.open(name, 'wb')
        return name if 'b' in mode else DADAStreamWriter(name, *args, **kwargs)
    elif 'r' in mode:
        if not hasattr(name, 'read'):
            name = io.open(name, 'rb')
        return name if 'b' in mode else DADAStreamReader(name, *args, **kwargs)
    else:
        raise ValueError("Only support opening DADA file for reading "
                         "or writing (mode='r' or 'w').")
