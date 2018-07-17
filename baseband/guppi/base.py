# Licensed under the GPLv3 - see LICENSE
from __future__ import division, unicode_literals, print_function

import re
import astropy.units as u
from astropy.utils import lazyproperty

from ..helpers import sequentialfile as sf
from ..vlbi_base.base import (make_opener, VLBIFileBase, VLBIFileReaderBase,
                              VLBIStreamBase, VLBIStreamReaderBase,
                              VLBIStreamWriterBase)
from .header import GUPPIHeader
from .payload import GUPPIPayload
from .frame import GUPPIFrame


__all__ = ['GUPPIFileNameSequencer', 'GUPPIFileReader', 'GUPPIFileWriter',
           'GUPPIStreamBase', 'GUPPIStreamReader', 'GUPPIStreamWriter', 'open']


class GUPPIFileNameSequencer(sf.FileNameSequencer):
    """List-like generator of GUPPI filenames using a template.

    The template is formatted, filling in any items in curly brackets with
    values from the header, as well as possibly a file number equal to the
    indexing value, indicated with '{file_nr}'.

    The length of the instance will be the number of files that exist that
    match the template for increasing values of the file number.

    Parameters
    ----------
    template : str
        Template to format to get specific filenames.  Curly bracket item
        keywords are not case-sensitive.
    header : dict-like
        Structure holding key'd values that are used to fill in the format.
        Keys must be in all caps (eg. ``DATE``), as with GUPPI header keys.

    Examples
    --------

    >>> from baseband import guppi
    >>> gfs = guppi.base.GUPPIFileNameSequencer(
    ...     '{date}_{file_nr:03d}.raw', {'DATE': "2018-01-01"})
    >>> gfs[10]
    '2018-01-01_010.raw'
    >>> from baseband.data import SAMPLE_PUPPI
    >>> with open(SAMPLE_PUPPI, 'rb') as fh:
    ...     header = guppi.GUPPIHeader.fromfile(fh)
    >>> template = 'puppi_{stt_imjd}_{src_name}_{scannum}.{file_nr:04d}.raw'
    >>> gfs = guppi.base.GUPPIFileNameSequencer(template, header)
    >>> gfs[0]
    'puppi_58132_J1810+1744_2176.0000.raw'
    >>> gfs[10]
    'puppi_58132_J1810+1744_2176.0010.raw'
    """
    def __init__(self, template, header):
        # convert template names to upper case, since header keywords are
        # upper case as well.
        self.items = {}

        def check_and_convert(x):
            string = x.group().upper()
            key = string[1:-1]
            if key != 'FILE_NR':
                self.items[key] = header[key]
            return string

        self.template = re.sub(r'{\w+[}:]', check_and_convert, template)

    def _process_items(self, file_nr):
        super(GUPPIFileNameSequencer, self)._process_items(file_nr)
        # Pop file_nr, as we need to capitalize it.
        self.items['FILE_NR'] = self.items.pop('file_nr')

    @classmethod
    def template_first_file(cls, name, **kwargs):
        """Returns the first file of a template based on its keys.

        Used by openers to access first header within a file sequence.
        """
        kwargs = {key.upper(): value for key, value in kwargs.items()}
        kwargs.setdefault('FILE_NR', 0)
        return cls(name, kwargs)[kwargs['FILE_NR']]


class GUPPIFileReader(VLBIFileReaderBase):
    """Simple reader for GUPPI files.

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
        header : `~baseband.guppi.GUPPIHeader`
        """
        return GUPPIHeader.fromfile(self.fh_raw)

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
        frame : `~baseband.guppi.GUPPIFrame`
            With ``.header`` and ``.payload`` properties.  The ``.data``
            property returns all data encoded in the frame.  Since this may
            be too large to fit in memory, it may be better to access the
            parts of interest by slicing the frame.
        """
        return GUPPIFrame.fromfile(self.fh_raw, memmap=memmap, verify=verify)

    def get_frame_rate(self):
        """Determine the number of frames per second.

        The routine uses the sample rate and number of samples per frame
        (excluding overlap) from the first header in the file.

        Returns
        -------
        frame_rate : `~astropy.units.Quantity`
            Frames per second.
        """
        oldpos = self.tell()
        self.seek(0)
        try:
            header = self.read_header()
            return (header.sample_rate /
                    (header.samples_per_frame - header.overlap)).to(u.Hz)
        finally:
            self.seek(oldpos)


class GUPPIFileWriter(VLBIFileBase):
    """Simple writer/mapper for GUPPI files.

    Adds `write_frame` and `memmap_frame` methods to the VLBI binary file
    wrapper.  The latter allows one to encode data in pieces, writing to disk
    as needed.
    """
    def write_frame(self, data, header=None, **kwargs):
        """Write a single frame (header plus payload).

        Parameters
        ----------
        data : `~numpy.ndarray` or `~baseband.guppi.GUPPIFrame`
            If an array, a ``header`` should be given, which will be used to
            get the information needed to encode the array, and to construct
            the GUPPI frame.
        header : `~baseband.guppi.GUPPIHeader`
            Can instead give keyword arguments to construct a header.  Ignored
            if ``data`` is a `~baseband.guppi.GUPPIFrame` instance.
        **kwargs
            If ``header`` is not given, these are used to initialize one.
        """
        if not isinstance(data, GUPPIFrame):
            data = GUPPIFrame.fromdata(data, header, **kwargs)
        return data.tofile(self.fh_raw)

    def memmap_frame(self, header=None, **kwargs):
        """Get frame by writing the header to disk and mapping its payload.

        The header is written to disk immediately, but the payload is mapped,
        so that it can be filled in pieces, by setting slices of the frame.

        Parameters
        ----------
        header : `~baseband.guppi.GUPPIHeader`
            Written to disk immediately.  Can instead give keyword arguments to
            construct a header.
        **kwargs
            If ``header`` is not given, these are used to initialize one.

        Returns
        -------
        frame: `~baseband.guppi.GUPPIFrame`
            By assigning slices to data, the payload can be encoded piecewise.
        """
        if header is None:
            header = GUPPIHeader.fromvalues(**kwargs)
        header.tofile(self.fh_raw)
        payload = GUPPIPayload.fromfile(self.fh_raw, memmap=True,
                                        header=header)
        return GUPPIFrame(header, payload)


class GUPPIStreamBase(VLBIStreamBase):
    """Base for GUPPI streams."""

    _sample_shape_maker = GUPPIPayload._sample_shape_maker

    def __init__(self, fh_raw, header0, squeeze=True, subset=(), verify=True):

        # GUPPI headers report their offsets using 'PKTIDX', the number of
        # unique UDP data packets (i.e. excluding overlap) written since the
        # start of observation.  'PKTSIZE' is the packet size in bytes.  Here
        # we calculate the packets per frame.
        self._packets_per_frame = (
            (header0.payload_nbytes - header0.overlap * header0._bpcs // 8) //
            header0['PKTSIZE'])

        # Set samples per frame to unique ones, excluding overlap.
        samples_per_frame = header0.samples_per_frame - header0.overlap

        super(GUPPIStreamBase, self).__init__(
            fh_raw=fh_raw, header0=header0, sample_rate=header0.sample_rate,
            samples_per_frame=samples_per_frame,
            unsliced_shape=header0.sample_shape, bps=header0.bps,
            complex_data=header0.complex_data, squeeze=squeeze, subset=subset,
            fill_value=0., verify=verify)

    # Overriding so the docstring indicates the exclusion of the overlap.
    samples_per_frame = property(VLBIStreamBase.samples_per_frame.fget,
                                 VLBIStreamBase.samples_per_frame.fset,
                                 doc=("Number of complete samples per frame, "
                                      "excluding overlap."))


class GUPPIStreamReader(GUPPIStreamBase, VLBIStreamReaderBase):
    """GUPPI format reader.

    Allows access to GUPPI files as a continuous series of samples.

    Parameters
    ----------
    fh_raw : filehandle
        Filehandle of the raw GUPPI stream.
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
        fh_raw = GUPPIFileReader(fh_raw)
        header0 = GUPPIHeader.fromfile(fh_raw)
        super(GUPPIStreamReader, self).__init__(fh_raw, header0,
                                                squeeze=squeeze,
                                                subset=subset, verify=verify)

    @lazyproperty
    def _last_header(self):
        """Header of the last file for this stream."""
        # Seek forward rather than backward, as last frame often has missing
        # bytes.
        nframes, fframe = divmod(self.fh_raw.seek(0, 2),
                                 self.header0.frame_nbytes)
        self.fh_raw.seek((nframes - 1) * self.header0.frame_nbytes)
        return self.fh_raw.read_header()

    def _read_frame(self, index):
        self.fh_raw.seek(index * self.header0.frame_nbytes)
        frame = self.fh_raw.read_frame(memmap=True, verify=self.verify)
        assert (frame.header['PKTIDX'] - self.header0['PKTIDX'] ==
                index * self._packets_per_frame)
        return frame


class GUPPIStreamWriter(GUPPIStreamBase, VLBIStreamWriterBase):
    """GUPPI format writer.

    Encodes and writes sequences of samples to file.

    Parameters
    ----------
    raw : filehandle
        For writing the header and raw data to storage.
    header0 : :class:`~baseband.guppi.GUPPIHeader`
        Header for the first frame, holding time information, etc.  Can instead
        give keyword arguments to construct a header (see ``**kwargs``).
    squeeze : bool, optional
        If `True` (default), `write` accepts squeezed arrays as input,
        and adds any dimensions of length unity.
    **kwargs
        If no header is given, an attempt is made to construct one from these.
        For a standard header, this would include the following.

    --- Header keywords : (see :meth:`~baseband.guppi.GUPPIHeader.fromvalues`)

    time : `~astropy.time.Time`
        Start time of the file.  Must have an integer number of seconds.
    sample_rate : `~astropy.units.Quantity`
        Number of complete samples per second, i.e. the rate at which each
        channel of each polarization is sampled.
    samples_per_frame : int
        Number of complete samples per frame.  Can alternatively give
        ``payload_nbytes``.
    payload_nbytes : int
        Number of bytes per payload.  Can alternatively give ``samples_per_frame``.
    offset : `~astropy.units.Quantity` or `~astropy.time.TimeDelta`, optional
        Time offset from the start of the whole observation (default: 0).
    npol : int, optional
        Number of polarizations (default: 1).
    nchan : int, optional
        Number of channels (default: 1).  For GUPPI, complex data is only allowed
        when nchan > 1.
    bps : int, optional
        Bits per elementary sample, i.e. per real or imaginary component for
        complex data (default: 8).
    """

    def __init__(self, fh_raw, header0=None, squeeze=True, **kwargs):
        if header0 is None:
            header0 = GUPPIHeader.fromvalues(**kwargs)
        assert header0.get('OVERLAP', 0) == 0, ("overlap must be 0 when "
                                                "writing GUPPI files.")
        fh_raw = GUPPIFileWriter(fh_raw)
        super(GUPPIStreamWriter, self).__init__(fh_raw, header0,
                                                squeeze=squeeze)

    def _make_frame(self, index):
        header = self.header0.copy()
        header.update(pktidx=self.header0['PKTIDX'] +
                      index * self._packets_per_frame)
        return self.fh_raw.memmap_frame(header)

    def _write_frame(self, frame, valid=True):
        assert frame is self._frame
        frame.valid = valid
        # Deleting frame flushes memmap'd data to disk.
        # (Of course, this gets deleted automatically when going out of
        # scope, and furthermore the link in self._frame will still exist
        # -- it only gets deleted in VLBIStreamWriter.write)
        del frame


open = make_opener('GUPPI', globals(), GUPPIFileNameSequencer,
                   ('squeeze', 'subset', 'verify'), ('squeeze',),
                   default_frames_per_file=128, doc="""
--- For reading a stream : (see `~baseband.guppi.base.GUPPIStreamReader`)

squeeze : bool, optional
    If `True` (default), remove any dimensions of length unity from
    decoded data.
subset : indexing object or tuple of objects, optional
    Specific components of the complete sample to decode (after possibly
    squeezing).  If a single indexing object is passed, it selects
    polarizations.  With a tuple, the first selects polarizations and the
    second selects channels.  If the tuple is empty (default), all
    components are read.

--- For writing a stream : (see `~baseband.guppi.base.GUPPIStreamWriter`)

header0 : `~baseband.guppi.GUPPIHeader`
    Header for the first frame, holding time information, etc.  Can instead
    give keyword arguments to construct a header (see ``**kwargs``).
squeeze : bool, optional
    If `True` (default), writer accepts squeezed arrays as input, and adds
    any dimensions of length unity.
frames_per_file : int, optional
    When writing to a sequence of files, sets the number of frames
    within each file.  Default: 128.
**kwargs
    If the header is not given, an attempt will be made to construct one
    with any further keyword arguments.  See
    :class:`~baseband.guppi.base.GUPPIStreamWriter`.

Returns
-------
Filehandle
    :class:`~baseband.guppi.base.GUPPIFileReader` or
    :class:`~baseband.guppi.base.GUPPIFileWriter` (binary), or
    :class:`~baseband.guppi.base.GUPPIStreamReader` or
    :class:`~baseband.guppi.base.GUPPIStreamWriter` (stream).

Notes
-----
For streams, one can also pass in a list of files, or equivalently a
`~baseband.helpers.sequentialfile` object (opened in 'rb' mode for reading or
'w+b' for writing).
""")
