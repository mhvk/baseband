# Licensed under the GPLv3 - see LICENSE
import io
import re
import operator
from functools import reduce

import astropy.units as u
from astropy.utils import lazyproperty

from ..helpers import sequentialfile as sf
from ..base.base import (
    FileBase,
    StreamReaderBase, StreamWriterBase,
    FileOpener, FileInfo)
from ..base.file_info import FileReaderInfo
from ..base.utils import lcm
from .header import DADAHeader
from .payload import DADAPayload
from .frame import DADAFrame


__all__ = ['DADAFileNameSequencer', 'DADAFileReader', 'DADAFileWriter',
           'DADAStreamBase', 'DADAStreamReader', 'DADAStreamWriter',
           'open', 'info']


class DADAFileNameSequencer(sf.FileNameSequencer):
    """List-like generator of DADA filenames using a template.

    The template is formatted, filling in any items in curly brackets with
    values from the header, as well as possibly a file number equal to the
    indexing value, indicated with '{file_nr}'.  The value '{obs_offset}' is
    treated specially, in being calculated using ``header['OBS_OFFSET'] +
    file_nr * header['FILE_SIZE']``, where ``header['FILE_SIZE']`` is the file
    size in bytes.

    The length of the instance will be the number of files that exist that
    match the template for increasing values of the file number (when writing,
    it is the number of files that have so far been generated).

    Parameters
    ----------
    template : str
        Template to format to get specific filenames.  Curly bracket item
        keywords are not case-sensitive.
    header : dict-like
        Structure holding key'd values that are used to fill in the format.
        Keys must be in all caps (eg. ``DATE``), as with DADA header keys.

    Examples
    --------

    >>> from baseband import dada
    >>> dfs = dada.DADAFileNameSequencer(
    ...     '{date}_{file_nr:03d}.dada', {'DATE': "2018-01-01"})
    >>> dfs[10]
    '2018-01-01_010.dada'
    >>> from baseband.data import SAMPLE_DADA
    >>> with open(SAMPLE_DADA, 'rb') as fh:
    ...     header = dada.DADAHeader.fromfile(fh)
    >>> template = '{utc_start}.{obs_offset:016d}.000000.dada'
    >>> dfs = dada.DADAFileNameSequencer(template, header)
    >>> dfs[0]
    '2013-07-02-01:37:40.0000006400000000.000000.dada'
    >>> dfs[1]
    '2013-07-02-01:37:40.0000006400064000.000000.dada'
    >>> dfs[10]
    '2013-07-02-01:37:40.0000006400640000.000000.dada'
    """

    def __init__(self, template, header={}):
        self.items = {}

        def check_and_convert(x):
            string = x.group().upper()
            key = string[1:-1]
            if key != 'FRAME_NR' and key != 'FILE_NR':
                self.items[key] = header[key]
            return string

        # This converts template names to upper case, since header keywords are
        # all upper case.
        self.template = re.sub(r'{\w+[}:]', check_and_convert, template)
        self._has_obs_offset = 'OBS_OFFSET' in self.items
        if self._has_obs_offset:
            self._obs_offset0 = self.items['OBS_OFFSET']
            self._file_size = header['FILE_SIZE']

    def _process_items(self, file_nr):
        super()._process_items(file_nr)
        # Pop file_nr, as we need to capitalize it.
        file_nr = self.items.pop('file_nr')
        self.items['FRAME_NR'] = self.items['FILE_NR'] = file_nr
        if self._has_obs_offset:
            self.items['OBS_OFFSET'] = (self._obs_offset0
                                        + file_nr * self._file_size)


class DADAFileReader(FileBase):
    """Simple reader for DADA files.

    Wraps a binary filehandle, providing methods to help interpret the data,
    such as `read_frame` and `get_frame_rate`. By default, frame payloads
    are mapped rather than fully read into physical memory.

    Parameters
    ----------
    fh_raw : filehandle
        Filehandle of the raw binary data file.
    """
    info = FileReaderInfo()

    def read_header(self):
        """Read a single header from the file.

        Returns
        -------
        header : `~baseband.dada.DADAHeader`
        """
        return DADAHeader.fromfile(self.fh_raw)

    def read_frame(self, memmap=True, verify=True):
        """Read the frame header and read or map the corresponding payload.

        Parameters
        ----------
        memmap : bool, optional
            If `True` (default), map the payload using `~numpy.memmap`, so that
            parts are only loaded into memory as needed to access data.
        verify : bool, optional
            Whether to do basic checks of frame integrity.  Default: `True`.

        Returns
        -------
        frame : `~baseband.dada.DADAFrame`
            With ``.header`` and ``.payload`` properties.  The ``.data``
            property returns all data encoded in the frame.  Since this may
            be too large to fit in memory, it may be better to access the
            parts of interest by slicing the frame.
        """
        return DADAFrame.fromfile(self.fh_raw, memmap=memmap, verify=verify)

    def get_frame_rate(self):
        """Determine the number of frames per second.

        The routine uses the sample rate and number of samples per frame
        from the first header in the file.

        Returns
        -------
        frame_rate : `~astropy.units.Quantity`
            Frames per second.
        """
        with self.temporary_offset(0):
            header = self.read_header()
        return (header.sample_rate / header.samples_per_frame).to(u.Hz)


class DADAFileWriter(FileBase):
    """Simple writer/mapper for DADA files.

    Adds `write_frame` and `memmap_frame` methods to the binary file wrapper.
    The latter allows one to encode data in pieces, writing to disk as needed.
    """

    def write_frame(self, data, header=None, **kwargs):
        """Write a single frame (header plus payload).

        Parameters
        ----------
        data : `~numpy.ndarray` or `~baseband.dada.DADAFrame`
            If an array, a ``header`` should be given, which will be used to
            get the information needed to encode the array, and to construct
            the DADA frame.
        header : `~baseband.dada.DADAHeader`
            Can instead give keyword arguments to construct a header.  Ignored
            if ``data`` is a `~baseband.dada.DADAFrame` instance.
        **kwargs
            If ``header`` is not given, these are used to initialize one.
        """
        if not isinstance(data, DADAFrame):
            data = DADAFrame.fromdata(data, header, **kwargs)
        return data.tofile(self.fh_raw)

    def memmap_frame(self, header=None, **kwargs):
        """Get frame by writing the header to disk and mapping its payload.

        The header is written to disk immediately, but the payload is mapped,
        so that it can be filled in pieces, by setting slices of the frame.

        Parameters
        ----------
        header : `~baseband.dada.DADAHeader`
            Written to disk immediately.  Can instead give keyword arguments to
            construct a header.
        **kwargs
            If ``header`` is not given, these are used to initialize one.

        Returns
        -------
        frame: `~baseband.dada.DADAFrame`
            By assigning slices to data, the payload can be encoded piecewise.
        """
        if header is None:
            header = DADAHeader.fromvalues(**kwargs)
        header.tofile(self.fh_raw)
        payload = DADAPayload.fromfile(self.fh_raw, memmap=True, header=header)
        return DADAFrame(header, payload)


class DADAStreamBase:
    """Provides sample shape maker and fast index getting/setting."""

    _sample_shape_maker = DADAPayload._sample_shape_maker

    def _get_index(self, header):
        # Override for faster calculation of frame index.
        return int(round((header['OBS_OFFSET']
                          - self.header0['OBS_OFFSET'])
                         / self.header0.payload_nbytes))

    def _set_index(self, header, index):
        header.update(obs_offset=self.header0['OBS_OFFSET']
                      + index * self.header0.payload_nbytes)


class DADAStreamReader(DADAStreamBase, StreamReaderBase):
    """DADA format reader.

    Allows access to DADA files as a continuous series of samples.

    Parameters
    ----------
    fh_raw : filehandle
        Filehandle of the raw DADA stream.
    squeeze : bool, optional
        If `True` (default), remove any dimensions of length unity from
        decoded data.
    subset : indexing object or tuple of objects, optional
        Specific components of the complete sample to decode (after possibly
        squeezing).  If a single indexing object is passed, it selects
        polarizations.  With a tuple, the first selects polarizations and the
        second selects channels.  If the tuple is empty (default), all
        components are read.
    verify : bool, optional
        Whether to do basic checks of frame integrity when reading.  The first
        frame of the stream is always checked, so ``verify`` is effective only
        when reading sequences of files.  Default: `True`.
    """

    def __init__(self, fh_raw, squeeze=True, subset=(), verify=True):
        fh_raw = DADAFileReader(fh_raw)
        header0 = fh_raw.read_header()
        super().__init__(fh_raw, header0, squeeze=squeeze, subset=subset,
                         verify=verify)
        # Store number of frames, for finding last header.
        with self.fh_raw.temporary_offset() as fh_raw:
            self._raw_file_size = fh_raw.seek(0, 2)
            self._nframes, partial_frame_nbytes = divmod(
                self._raw_file_size, self.header0.frame_nbytes)
            # If there is a partial last frame.
            if partial_frame_nbytes > 0:
                # If partial last frame contains payload bytes.
                if partial_frame_nbytes > self.header0.nbytes:
                    self._nframes += 1
                    # If there's only one frame and it's incomplete.
                    if self._nframes == 1:
                        self._header0 = self._last_header
                        self._samples_per_frame = (
                            self._last_header.samples_per_frame)
                # Otherwise, ignore the partial frame unless it's the only
                # frame, in which case raise an EOFError.
                elif self._nframes == 0:
                    raise EOFError('file (of {0} bytes) appears to end without'
                                   'any payload.'.format(partial_frame_nbytes))

    @lazyproperty
    def _last_header(self):
        """Header of the last file for this stream.

        If last frame is prematurely truncated, header's payload_nbytes is
        reduced accordingly to let the stream reader read to the end of file.
        """
        # We assume DADA files are complete except for possibly the last one.
        # Hence, we seek directly to where the last header should be, but
        # adjust it if the last file is short.
        with self.fh_raw.temporary_offset() as fh_raw:
            self._seek_frame(self._nframes - 1)
            header = fh_raw.read_header()
            payload_nbytes = self._raw_file_size - fh_raw.tell()
            assert payload_nbytes > 0, 'setup failed: no payload in last frame'
            if header.payload_nbytes > payload_nbytes:
                # Truncated last frame.  Adjust header to give the actual
                # number of useful samples, insisting that the payload has
                # integer number of both words and complete samples.
                header.mutable = True
                payload_block = lcm(
                    DADAPayload._dtype_word.itemsize,
                    reduce(operator.mul, self.sample_shape,
                           self.header0.bps * (2 if self.header0.complex_data
                                               else 1) // 8))
                header.payload_nbytes = ((payload_nbytes // payload_block)
                                         * payload_block)
                header.mutable = False

        return header

    @lazyproperty
    def stop_time(self):
        """Time at the end of the file, just after the last sample.

        See also `start_time` for the start time of the file, and `time` for
        the time of the sample pointer's current offset.
        """
        return (self._get_time(self._last_header)
                + (self._last_header.samples_per_frame
                   / self.sample_rate).to(u.s))

    def _fh_raw_read_frame(self):
        # Override to special-case last frame, which may be short.
        if (self.fh_raw.tell() // self.header0.frame_nbytes
                < self._nframes - 1):
            return self.fh_raw.read_frame()

        # Use last header, which will have properly adjusted payload size.
        self.fh_raw.seek(self.header0.nbytes, 1)
        last_payload = DADAPayload.fromfile(self.fh_raw, memmap=True,
                                            header=self._last_header)
        # Ensure we skip all the way to the end of the file, to indicate
        # there is no use in trying to check for the next header.
        self.fh_raw.seek(0, 2)
        return DADAFrame(self._last_header, last_payload)


class DADAStreamWriter(DADAStreamBase, StreamWriterBase):
    """DADA format writer.

    Encodes and writes sequences of samples to file.

    Parameters
    ----------
    raw : filehandle
        For writing the header and raw data to storage.
    header0 : :class:`~baseband.dada.DADAHeader`
        Header for the first frame, holding time information, etc.
    squeeze : bool, optional
        If `True` (default), `write` accepts squeezed arrays as input,
        and adds any dimensions of length unity.
    """

    def __init__(self, fh_raw, header0, squeeze=True):
        assert header0.get('OBS_OVERLAP', 0) == 0
        fh_raw = DADAFileWriter(fh_raw)
        super().__init__(fh_raw, header0, squeeze=squeeze)

    def _make_frame(self, index):
        header = self.header0.copy()
        self._set_index(header, index)
        return self.fh_raw.memmap_frame(header)

    def _fh_raw_write_frame(self, frame):
        assert frame is self._frame
        # Deleting frame flushes memmap'd data to disk.
        del self._frame


class DADAFileOpener(FileOpener):
    FileNameSequencer = DADAFileNameSequencer

    def get_fns(self, name, mode, kwargs):
        fns = super().get_fns(name, mode, kwargs)
        # For obs_offset we need the first file to know the
        # actual file_size.
        if mode[0] == 'r' and 'obs_offset' in name.lower():
            with io.open(fns[0], 'rb') as fh:
                header0 = DADAHeader.fromfile(fh)
            fns = self.FileNameSequencer(name, header0)
        return fns

    def get_fh(self, name, mode, kwargs):
        if mode == 'ws' and self.is_sequence(name):
            kwargs.setdefault('file_size', kwargs['header0'].frame_nbytes)

        return super().get_fh(name, mode, kwargs)


open = DADAFileOpener.create(globals(), doc="""
--- For reading a stream : (see :class:`~baseband.dada.base.DADAStreamReader`)

squeeze : bool, optional
    If `True` (default), remove any dimensions of length unity from
    decoded data.
subset : indexing object or tuple of objects, optional
    Specific components of the complete sample to decode (after possibly
    squeezing).  If a single indexing object is passed, it selects
    polarizations.  With a tuple, the first selects polarizations and the
    second selects channels.  If the tuple is empty (default), all
    components are read.

--- For writing a stream : (see :class:`~baseband.dada.base.DADAStreamWriter`)

header0 : `~baseband.dada.DADAHeader`
    Header for the first frame, holding time information, etc.  Can instead
    give keyword arguments to construct a header (see ``**kwargs``).
squeeze : bool, optional
    If `True` (default), writer accepts squeezed arrays as input, and adds
    any dimensions of length unity.
**kwargs
    If no header is given, an attempt is made to construct one from these.
    For a standard header, this would include the following.

--- Header keywords : (see :meth:`~baseband.dada.DADAHeader.fromvalues`)

time : `~astropy.time.Time`
    Start time of the file.
samples_per_frame : int,
    Number of complete samples per frame.
sample_rate : `~astropy.units.Quantity`
    Number of complete samples per second, i.e. the rate at which each
    channel of each polarization is sampled.
offset : `~astropy.units.Quantity` or `~astropy.time.TimeDelta`, optional
    Time offset from the start of the whole observation (default: 0).
npol : int, optional
    Number of polarizations (default: 1).
nchan : int, optional
    Number of channels (default: 1).
complex_data : bool, optional
    Whether data are complex (default: `False`).
bps : int, optional
    Bits per elementary sample, i.e. per real or imaginary component for
    complex data (default: 8).

Returns
-------
Filehandle
    :class:`~baseband.dada.base.DADAFileReader` or
    :class:`~baseband.dada.base.DADAFileWriter` (binary), or
    :class:`~baseband.dada.base.DADAStreamReader` or
    :class:`~baseband.dada.base.DADAStreamWriter` (stream).

Notes
-----
For streams, one can also pass to ``name`` a list of files, or a template
string that can be formatted using 'frame_nr', 'obs_offset', and other header
keywords (by `~baseband.dada.DADAFileNameSequencer`).

For writing, one can mimic what is done at quite a few telescopes by using
the template '{utc_start}_{obs_offset:016d}.000000.dada'.  Unlike for most
openers, ``file_size`` is set to the size of one frame as given by the header.

For reading, to read series such as the above, use something like
'2013-07-02-01:37:40_{obs_offset:016d}.000000.dada'.  Note that here we
have to pass in the date explicitly, since the template is used to get the
first file name, before any header is read, and therefore the only keywords
available are 'frame_nr', 'file_nr', and 'obs_offset', all of which are
assumed to be zero for the first file. To avoid this restriction, pass in
keyword arguments with values appropriate for the first file.

One may also pass in a `~baseband.helpers.sequentialfile` object
(opened in 'rb' mode for reading or 'w+b' for writing), though for typical use
cases it is practically identical to passing in a list or template.
""")


info = FileInfo.create(globals())
