# Licensed under the GPLv3 - see LICENSE.rst
import os
import io
import re

import numpy as np
from astropy.extern import six
from astropy.utils import lazyproperty

from ..vlbi_base.base import (VLBIFileBase, VLBIStreamBase,
                              VLBIStreamReaderBase, VLBIStreamWriterBase)
from .header import DADAHeader
from .payload import DADAPayload
from .frame import DADAFrame


__all__ = ['DADAFileNameSequencer', 'DADAFileReader', 'DADAFileWriter',
           'DADAStreamBase', 'DADAStreamReader', 'DADAStreamWriter', 'open']


class DADAFileNameSequencer:
    """List-like generator of filenames using a template.

    The template is formatted, filling in any items in curly brackets with
    values from the header, as well as possibly a file number, indicated with
    '{file_nr}'.  The value '{obs_offset}' is treated specially, in being
    calculated using ``header['OBS_OFFSET'] + file_nr * header['FILE_SIZE']``.

    The length of the instance will be the number of files that exist that
    match the template for increasing values of the file fumber.

    Parameters
    ----------
    template : str
        Template to format to get specific filenames.
    header : dict-like
        Structure holding key'd values that are used to fill in the format.

    Examples
    --------

    >>> from baseband import dada
    >>> dfs = dada.base.DADAFileNameSequencer('a{file_nr:03d}.dada', {})
    >>> dfs[10]
    'a010.dada'
    >>> from baseband.data import SAMPLE_DADA
    >>> with open(SAMPLE_DADA, 'rb') as fh:
    ...     header = dada.DADAHeader.fromfile(fh)
    >>> template = '{utc_start}.{obs_offset:016d}.000000.dada'
    >>> dfs = DADAFileNameSequencer(template, header)
    >>> dfs[0]
    '2013-07-02-01:37:40.0000006400000000.000000.dada'
    >>> dfs[1]
    '2013-07-02-01:37:40.0000006400064000.000000.dada'
    >>> dfs[10]
    '2013-07-02-01:37:40.0000006400640000.000000.dada'
    """
    def __init__(self, template, header):
        # convert template names to upper case, since header keywords are
        # upper case as well.
        self.items = {}

        def check_and_convert(x):
            string = x.group().upper()
            key = string[1:-1]
            if key != 'FRAME_NR' and key != 'FILE_NR':
                self.items[key] = header[key]
            return string

        self.template = re.sub(r'{\w+[}:]', check_and_convert, template)
        self._has_obs_offset = 'OBS_OFFSET' in self.items
        if self._has_obs_offset:
            self._obs_offset0 = self.items['OBS_OFFSET']
            self._file_size = header['FILE_SIZE']

    def __getitem__(self, frame_nr):
        if frame_nr < 0:
            frame_nr += len(self)
            if frame_nr < 0:
                raise IndexError('frame number out of range.')

        self.items['FRAME_NR'] = self.items['FILE_NR'] = frame_nr
        if self._has_obs_offset:
            self.items['OBS_OFFSET'] = (self._obs_offset0 +
                                        frame_nr * self._file_size)
        return self.template.format(**self.items)

    def __len__(self):
        frame_nr = 0
        while os.path.isfile(self[frame_nr]):
            frame_nr += 1

        return frame_nr


class DADAFileReader(VLBIFileBase):
    """Simple reader for DADA files.

    Adds a ``read_frame`` method to the basic VLBI binary file wrapper. By
    default, the payload is mapped rather than fully read into physical memory.
    """
    def read_frame(self, memmap=True):
        """Read the frame header and read or map the corresponding payload.

        Parameters
        ----------
        memmap : bool, optional
            If `True` (default), map the payload using `~numpy.memmap`, so that
            parts are only loaded into memory as needed to access data.

        Returns
        -------
        frame : `~baseband.dada.DADAFrame`
            With ``.header`` and ``.payload`` properties.  The ``.data``
            property returns all data encoded in the frame.  Since this may
            be too large to fit in memory, it may be better to access the
            parts of interest by slicing the frame.
        """
        return DADAFrame.fromfile(self.fh, memmap=memmap)


class DADAFileWriter(VLBIFileBase):
    """Simple writer/mapper for DADA files.

    Adds ``write_frame`` and ``memmap_frame`` methods to the VLBI binary file
    wrapper.  The latter allows one to encode data in pieces, writing to disk
    as needed.
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
            Ignored if ``data`` is a DADA Frame.
        **kwargs
            Used to initialize a header if none was passed in explicitly.
        """
        if not isinstance(data, DADAFrame):
            data = DADAFrame.fromdata(data, header, **kwargs)
        return data.tofile(self.fh)

    def memmap_frame(self, header=None, **kwargs):
        """Get frame by writing the header to disk and mapping its payload.

        The header is written to disk immediately, but the payload is mapped,
        so that it can be filled in pieces, by setting slices of the frame.

        Parameters
        ----------
        header : `~baseband.dada.DADAHeader`, optional
            Written to disk immediately.
        **kwargs
            Used to initialize a header if none was passed in explicitly.

        Returns
        -------
        frame: `~baseband.dada.DADAFrame`
            By assigning slices to data, the payload can be encoded piecewise.
        """
        if header is None:
            header = DADAHeader.fromvalues(**kwargs)
        header.tofile(self.fh)
        payload = DADAPayload.fromfile(self.fh, memmap=True, header=header)
        return DADAFrame(header, payload)


class DADAStreamBase(VLBIStreamBase):
    """DADA file wrapper, which combines threads into streams.

    Parameters
    ----------
    raw : filehandle
        File handle of the (first) raw DADA stream.  Ignored if ``files`` or
        ``template`` is given.
    header0 : `~baseband.dada.DADAHeader`
        Header for the first frame, which is used to infer frame size,
        encoding, etc.
    thread_ids : list of int, optional
        Specific threads to use.  By default, as many as there are
        polarisations ('header0["NPOL"]').
    files : list or tuple of str, optional
        Should contain the names of all files of a given observation,
        in time-order.
    template : str, optional
        A template string that can be formatted using 'frame_nr', 'obs_offset',
        and other header keywords.  For details, see
        :class:`~baseband.dada.base.DADAFileNameSequencer`.
    """

    def __init__(self, fh_raw, header0, thread_ids=None, files=None,
                 template=None):
        frames_per_second = (1. / (header0['TSAMP'] * 1e-6) /
                             header0.samples_per_frame)
        if thread_ids is None:
            thread_ids = list(range(header0['NPOL']))
        super(DADAStreamBase, self).__init__(
            fh_raw=fh_raw, header0=header0, nchan=header0['NCHAN'],
            bps=header0.bps, complex_data=header0.complex_data,
            thread_ids=thread_ids,
            samples_per_frame=header0.samples_per_frame,
            frames_per_second=frames_per_second)
        if files and template:
            raise TypeError('cannot pass in both template and file list.')

        if template:
            self.files = DADAFileNameSequencer(template, header0)
        else:
            self.files = files
        self._frame_nr = None

    def _frame_info(self):
        """Convert offset to file number and offset into that file."""
        return divmod(self.offset, self.samples_per_frame)

    def _open_file(self, frame_nr, mode):
        if self.fh_raw:
            self.fh_raw.close()
        self.fh_raw = open(self.files[frame_nr], mode)

    def _unsqueeze(self, data):
        if data.ndim < 3 and self.nthread == 1:
            data = data[:, np.newaxis]
        if data.ndim < 3 and self.nchan == 1:
            data = data[..., np.newaxis]
        return data


class DADAStreamReader(DADAStreamBase, VLBIStreamReaderBase):
    """DADA format stream reader.

    This wrapper allows one to access a DADA file as a continues series of
    samples.

    Parameters
    ----------
    raw : filehandle
        file handle of the (first) raw DADA stream.
    thread_ids : list of int, optional
        Specific threads to read.  By default, all threads are read.
    files : list or tuple of str, optional
        Should contain the names of all files of a given observation,
        in time-order.
    template : str, optional
        A template string that can be formatted using 'frame_nr', 'obs_offset',
        and other header keywords.  Many series of dada files can be read with
        something like '2013-07-02-01:37:40_{obs_offset:016d}.000000.dada'.
        For details, see :class:`~baseband.dada.base.DADAFileNameSequencer`.
    """
    def __init__(self, raw, thread_ids=None, files=None, template=None):
        header = DADAHeader.fromfile(raw)
        super(DADAStreamReader, self).__init__(raw, header, thread_ids,
                                               files=files, template=template)
        if self.files is None:
            self._frame = DADAFrame(header, DADAPayload.fromfile(raw, header,
                                                                 memmap=True))
            self._frame_nr = 0
        else:
            self._get_frame(0)

    @lazyproperty
    def header1(self):
        """Header of the last file for this stream."""
        if self.files:
            self._get_frame(len(self.files) - 1)
        return self._frame.header

    def read(self, count=None, squeeze=True, out=None):
        """Read count samples.

        Parameters
        ----------
        count : int, optional
            Number of samples to read.  If omitted or negative, the whole
            file is read. Ignored if ``out`` is given.
        squeeze : bool
            If `True` (default), remove channel and thread dimensions if unity
            (or allow those to have been removed in ``out``).
        out : `None` or array
            Array to store the data in. If given, ``count`` will be inferred.
            If ``squeeze`` is `True`, unity dimensions can be absent.

        Returns
        -------
        out : array of float or complex
            Dimensions are (sample-time, thread, channel), with dimensions of
            length unity possibly removed (if ``squeeze`` is `True`, or if they
            were not present in the ``out`` array passed in).
        """
        if out is None:
            if count is None or count < 0:
                count = self.size - self.offset

            result = np.empty((count,) + self._frame.sample_shape,
                              dtype=self._frame.dtype)
            # Generate view of the result data set that will be returned.
            out = result.squeeze() if squeeze else result
        else:
            count = out.shape[0]
            # Create a properly-shaped view of the output if needed.
            result = self._unsqueeze(out) if squeeze else out

        offset0 = self.offset
        while count > 0:
            frame_nr, sample_offset = self._frame_info()
            if frame_nr != self._frame_nr:
                # Open relevant file.
                self._get_frame(frame_nr)

            # Copy relevant data from frame into output.
            nsample = min(count, self.samples_per_frame - sample_offset)
            sample = self.offset - offset0
            data_slice = slice(sample_offset, sample_offset + nsample)
            if self.thread_ids:
                data_slice = (data_slice, self.thread_ids)

            result[sample:sample + nsample] = self._frame[data_slice]
            self.offset += nsample
            count -= nsample

        return out

    def _get_frame(self, frame_nr):
        self._open_file(frame_nr, 'rb')
        self._frame = self.fh_raw.read_frame(memmap=True)
        assert (self._frame.header['OBS_OFFSET'] ==
                self.header0['OBS_OFFSET'] + frame_nr *
                self.header0.payloadsize)


class DADAStreamWriter(DADAStreamBase, VLBIStreamWriterBase):
    """DADA format writer.

    Parameters
    ----------
    raw : filehandle
        For writing the header and raw data.  This can be `None` if ``files``
        or ``template`` is passed in.
    header : :class:`~baseband.dada.DADAHeader`, optional
        Header for the file, holding time information, etc.
    files : list or tuple of str, optional
        Should contain the names of all files to be used to write the data,
        in time-order.
    template : str, optional
        A template string that can be formatted using 'frame_nr', 'obs_offset',
        and other header keywords.  To reproduce what is used by some other
        implementations, use '{utc_start}_{obs_offset:016d}.000000.dada'.
        For details, see :class:`~baseband.dada.base.DADAFileNameSequencer`.
    **kwargs
        Used to construct a header if none is passed in explicitly.
        For a standard header, this would include the following.

    --- Header keywords : (see :meth:`~baseband.dada.DADAHeader.fromvalues`)

    time : `~astropy.time.Time`
        The start time of this file.
    offset : `~astropy.units.Quantity`
        A possible time offset from the start of the whole observation.
    npol : int, optional
        Number of polarizations (and thus threads; default 1).
    nchan : int, optional
        Number of FFT channels within stream (default 1).
        Note: that different # of channels per thread is not supported.
    complex_data : bool
        Whether data is complex
    bps : int
        Bits per sample (or real, imaginary component).
    samples_per_frame : int
        Number of complete samples in a given frame.
    bandwidth : `~astropy.units.Quantity`
        Bandwidth spanned by the data.  Used to infer the sample rate and
        thus to calculate times.
    """
    def __init__(self, raw, header=None, files=None, template=None, **kwargs):
        if header is None:
            if 'nthread' in kwargs:
                kwargs.setdefault('npol', kwargs.pop('nthread'))
            header = DADAHeader.fromvalues(**kwargs)
        assert header.get('OBS_OVERLAP', 0) == 0
        super(DADAStreamWriter, self).__init__(raw, header, files=files,
                                               template=template)
        if self.files is None:
            self._frame = self.fh_raw.memmap_frame(header)
            self._frame_nr = 0
        else:
            self._get_frame(0)

    def write(self, data, squeezed=True, invalid_data=False):
        """Write data, using multiple files as needed.

        Parameters
        ----------
        data : array
            Piece of data to be written.  This should be properly scaled to
            make best use of the dynamic range delivered by the encoding.
        squeezed : bool, optional
            If `True` (default), allow dimensions with size one to be absent
            (i.e., if ``nthread`` and/or ``nchan`` is unity).
        invalid_data : bool
            Whether the current data is valid.  Present for consistency with
            other stream writers.  It does not seem possible to store this
            information in dada files.
        """
        if squeezed:
            data = self._unsqueeze(data)

        assert data.shape[1] == self.nthread
        assert data.shape[2] == self.nchan

        count = data.shape[0]
        sample = 0
        offset0 = self.offset
        while count > 0:
            frame_nr, sample_offset = self._frame_info()
            if self._frame_nr is None:
                assert sample_offset == 0
                self._get_frame(frame_nr)

            nsample = min(count, self.samples_per_frame - sample_offset)
            sample_end = sample_offset + nsample
            sample = self.offset - offset0
            self._frame[sample_offset:
                        sample_end] = data[sample:sample + nsample]
            if sample_end == self.samples_per_frame:
                # deleting frame flushes memmap'd data to disk
                del self._frame
                self._frame_nr = None

            self.offset += nsample
            count -= nsample

    def _get_frame(self, frame_nr):
        self._open_file(frame_nr, 'wb')
        # set up header for new frame.
        header = self.header0.copy()
        header.update(obs_offset=self.header0['OBS_OFFSET'] +
                      frame_nr * self.header0.payloadsize)
        self._frame = self.fh_raw.memmap_frame(header)
        self._frame_nr = frame_nr


def open(name, mode='rs', *args, **kwargs):
    """Open DADA format file for reading or writing.

    Opened as binary, one gets a standard file handle.  Opened as a stream, the
    file handle is wrapped, allowing access to it as a series of samples.

    For streams, one can also pass in a list of files or a template string that
    can be formatted using 'frame_nr', 'obs_offset', and other header keywords
    (using :class:`~baseband.dada.base.DADAFileNameSequencer`).

    For writing, one can mimic what is done at quite a few telescopes by using
    the template '{utc_start}_{obs_offset:016d}.000000.dada'.

    For reading, to read series such as the above, use something like
    '2013-07-02-01:37:40_{obs_offset:016d}.000000.dada'.  Note that here we
    have to pass in the date explicitly, since the template is used to get the
    first file name, before any header is read, and therefore the only keywords
    available are 'frame_nr', 'file_nr', and 'obs_offset', all of which are
    assumed to be zero for the first file. To avoid this restriction, pass in
    the first file name directly, and use the ``template`` keyword argument.

    Parameters
    ----------
    name : str, list or tuple of str
        File name or name template, or series of file names.
    mode : {'rb', 'wb', 'rs', or 'ws'}, optional
        Whether to open for reading or writing, and as a regular binary file
        or as a stream (default is reading a stream).
    *args, **kwargs
        Additional arguments when opening the file as a stream.

    --- For reading : (see :class:`~baseband.dada.base.DADAStreamReader`)

    thread_ids : list of int, optional
        Specific threads to read.  By default, all threads are read.
        (For DADA, the number threads equals ``header['NPOL']``, i.e.,
        the number of polarisations.)

    --- For writing : (see :class:`~baseband.dada.base.DADAStreamWriter`)

    header : `~baseband.dada.DADAHeader`, optional
        Header for the first frame, holding time information, etc.
    **kwargs
        If the header is not given, an attempt will be made to construct one
        with any further keyword arguments.  See
        :class:`~baseband.dada.base.DADAStreamWriter`.

    Returns
    -------
    Filehandle
        :class:`~baseband.dada.base.DADAFileReader` or
        :class:`~baseband.dada.base.DADAFileWriter` instance (binary), or
        :class:`~baseband.dada.base.DADAStreamReader` or
        :class:`~baseband.dada.base.DADAStreamWriter` instance (stream).
    """
    if 'b' not in mode:
        if isinstance(name, (tuple, list)):
            kwargs['files'] = name
            name = name[0]
        elif isinstance(name, six.string_types) and ('{' in name and
                                                     '}' in name):
            kwargs['template'] = name
            if 'w' in mode:
                return DADAStreamWriter(None, *args, **kwargs)
            try:
                name = name.format(frame_nr=0, file_nr=0, obs_offset=0)
            except KeyError:
                raise KeyError(
                    "For reading, a template for file names can only contain "
                    "'file_nr', 'frame_nr', or 'obs_offset', since the header "
                    "is not available for the first file. One can pass in a "
                    "full file name and use 'template={0}'".format(name))

    if 'w' in mode:
        if not hasattr(name, 'write'):
            name = io.open(name, 'w+b')
        try:
            fh = DADAFileWriter(name)
            return fh if 'b' in mode else DADAStreamWriter(fh, *args, **kwargs)
        except Exception:
            name.close()
            raise
    elif 'r' in mode:
        if not hasattr(name, 'read'):
            name = io.open(name, 'rb')
        try:
            fh = DADAFileReader(name)
            return fh if 'b' in mode else DADAStreamReader(fh, *args, **kwargs)
        except Exception:
            name.close()
            raise
    else:
        raise ValueError("Only support opening DADA file for reading "
                         "or writing (mode='r' or 'w').")
