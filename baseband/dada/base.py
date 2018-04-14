# Licensed under the GPLv3 - see LICENSE
from __future__ import division, unicode_literals, print_function
import io
import os
import re

import numpy as np
import astropy.units as u
from astropy.extern import six
from astropy.utils import lazyproperty

from ..helpers import sequentialfile as sf
from ..vlbi_base.base import (make_opener, VLBIFileBase, VLBIStreamBase,
                              VLBIStreamReaderBase, VLBIStreamWriterBase)
from .header import DADAHeader
from .payload import DADAPayload
from .frame import DADAFrame


__all__ = ['DADAFileNameSequencer', 'DADAFileReader', 'DADAFileWriter',
           'DADAStreamBase', 'DADAStreamReader', 'DADAStreamWriter', 'open']


class DADAFileNameSequencer:
    """List-like generator of filenames using a template.

    The template is formatted, filling in any items in curly brackets with
    values from the header, as well as possibly a file number equal to the
    indexing value, indicated with '{file_nr}'.  The value '{obs_offset}' is
    treated specially, in being calculated using ``header['OBS_OFFSET'] +
    file_nr * header['FILE_SIZE']``, where ``header['FILE_SIZE']`` is the file
    size in bytes.

    The length of the instance will be the number of files that exist that
    match the template for increasing values of the file number.

    Parameters
    ----------
    template : str
        Template to format to get specific filenames.
    header : dict-like
        Structure holding key'd values that are used to fill in the format.
        Keys must be in all caps (eg. ``DATE``), as with DADA header keys.

    Examples
    --------

    >>> from baseband import dada
    >>> dfs = dada.base.DADAFileNameSequencer('a{file_nr:03d}.dada', {})
    >>> dfs[10]
    'a010.dada'
    >>> dfs = dada.base.DADAFileNameSequencer(
    ...     '{date}_{file_nr:03d}.dada', {'DATE': "2018-01-01"})
    >>> dfs[10]
    '2018-01-01_010.dada'
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

    Wraps a binary filehandle, providing methods to help interpret the data,
    such as `read_frame` and `get_frame_rate`. By default, frame payloads
    are mapped rather than fully read into physical memory.

    Parameters
    ----------
    fh_raw : filehandle
        Filehandle of the raw binary data file.
    """
    def read_header(self):
        """Read a single header from the file.

        Returns
        -------
        header : `~baseband.dada.DADAHeader`
        """
        return DADAHeader.fromfile(self.fh_raw)

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
        return DADAFrame.fromfile(self.fh_raw, memmap=memmap)

    def get_frame_rate(self):
        """Determine the number of frames per second.

        The routine uses the sample rate and number of samples per frame
        from the first header in the file.

        Returns
        -------
        frame_rate : `~astropy.units.Quantity`
            Frames per second.
        """
        oldpos = self.tell()
        self.seek(0)
        try:
            header = self.read_frame()
            return header.sample_rate / header.samples_per_frame
        finally:
            self.seek(oldpos)


class DADAFileWriter(VLBIFileBase):
    """Simple writer/mapper for DADA files.

    Adds `write_frame` and `memmap_frame` methods to the VLBI binary file
    wrapper.  The latter allows one to encode data in pieces, writing to disk
    as needed.
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


class DADAStreamBase(VLBIStreamBase):
    """Base for DADA streams."""

    _sample_shape_maker = DADAPayload._sample_shape_maker

    def __init__(self, fh_raw, header0, squeeze=True, subset=()):

        super(DADAStreamBase, self).__init__(
            fh_raw=fh_raw, header0=header0, sample_rate=header0.sample_rate,
            samples_per_frame=header0.samples_per_frame,
            unsliced_shape=header0.sample_shape, bps=header0.bps,
            complex_data=header0.complex_data, squeeze=squeeze, subset=subset,
            fill_value=0.)


class DADAStreamReader(DADAStreamBase, VLBIStreamReaderBase):
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
    """
    def __init__(self, fh_raw, squeeze=True, subset=()):
        fh_raw = DADAFileReader(fh_raw)
        header0 = DADAHeader.fromfile(fh_raw)
        super(DADAStreamReader, self).__init__(fh_raw, header0,
                                               squeeze=squeeze, subset=subset)

    @lazyproperty
    def _last_header(self):
        """Header of the last file for this stream."""
        self.fh_raw.seek(-self.header0.frame_nbytes, 2)
        last_frame = self.fh_raw.read_frame(memmap=True)
        return last_frame.header

    def _read_frame(self, index):
        self.fh_raw.seek(index * self.header0.frame_nbytes)
        frame = self.fh_raw.read_frame(memmap=True)
        assert (frame.header['OBS_OFFSET'] - self.header0['OBS_OFFSET'] ==
                index * self.header0.payload_nbytes)
        return frame


class DADAStreamWriter(DADAStreamBase, VLBIStreamWriterBase):
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
        super(DADAStreamWriter, self).__init__(fh_raw, header0,
                                               squeeze=squeeze)

    def _make_frame(self, index):
        header = self.header0.copy()
        header.update(obs_offset=self.header0['OBS_OFFSET'] +
                      index * self.header0.payload_nbytes)
        return self.fh_raw.memmap_frame(header)

    def _write_frame(self, frame, valid=True):
        assert frame is self._frame
        frame.valid = valid
        # Deleting frame flushes memmap'd data to disk.
        # (Of course, this gets deleted automatically when going out of
        # scope, and furthermore the link in self._frame will still exist
        # -- it only gets deleted in VLBIStreamWriter.write)
        del frame


opener = make_opener('DADA', globals(), doc="""
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
    If the header is not given, an attempt will be made to construct one
    with any further keyword arguments.

--- Header keywords : (see :meth:`~baseband.dada.DADAHeader.fromvalues`)

time : `~astropy.time.Time`
    Start time of the file.
samples_per_frame : int,
    Number of complete samples per frame.
sample_rate : `~astropy.units.Quantity`
    Number of complete samples per second, i.e. the rate at which each
    channel of each polarization is sampled.
offset : `~astropy.units.Quantity`, optional
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
""")


# Need to wrap the opener to be able to deal with file lists or templates.
# TODO: move this up to the opener??
def open(name, mode='rs', **kwargs):
    header0 = kwargs.get('header0', None)
    squeeze = kwargs.pop('squeeze', None)
    # If sequentialfile object, check that it's opened properly.
    if isinstance(name, sf.SequentialFileBase):
        assert (('r' in mode and name.mode == 'rb') or
                ('w' in mode and name.mode == 'w+b')), (
                    "open only accepts sequential files opened in 'rb' mode "
                    "for reading or 'w+b' mode for writing.")
    is_template = isinstance(name, six.string_types) and ('{' in name and
                                                          '}' in name)
    is_sequence = isinstance(name, (tuple, list))

    if 'b' not in mode:
        if header0 is None:
            if 'w' in mode:
                # For writing a header is required.
                header0 = DADAHeader.fromvalues(**kwargs)
                kwargs = {}

            elif is_template and ('OBS_OFFSET' in name or
                                  'obs_offset' in name):
                # For reading try reading header from first file if needed.
                # We make a temporary file sequencer for this, as the real one
                # will need the header file size.
                kwargs = {key.upper(): value for key, value in kwargs.items()}
                for key in ('FILE_NR', 'FRAME_NR', 'OBS_OFFSET', 'FILE_SIZE'):
                    kwargs.setdefault(key, 0)
                first_file = (DADAFileNameSequencer(name, kwargs)
                              [kwargs['FRAME_NR']])
                with io.open(first_file, 'rb') as fh:
                    header0 = DADAHeader.fromfile(fh)
                kwargs = {}
            else:
                header0 = {}

        if is_template:
            name = DADAFileNameSequencer(name, header0)

        if is_template or is_sequence:
            if 'r' in mode:
                name = sf.open(name, 'rb')
            else:
                name = sf.open(name, 'w+b', file_size=header0.frame_nbytes)

        if header0 and 'w' in mode:
            kwargs['header0'] = header0

    if squeeze is not None:
        kwargs['squeeze'] = squeeze

    return opener(name, mode, **kwargs)


open.__doc__ = opener.__doc__ + """\n
Notes
-----
For streams, one can also pass in a list of files, or a template string that
can be formatted using 'frame_nr', 'obs_offset', and other header keywords
(by `~baseband.dada.base.DADAFileNameSequencer`).

For writing, one can mimic what is done at quite a few telescopes by using
the template '{utc_start}_{obs_offset:016d}.000000.dada'.

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
"""
