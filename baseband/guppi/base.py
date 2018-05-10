# Licensed under the GPLv3 - see LICENSE
import os
import io
import re

import numpy as np
import astropy.units as u
from astropy.extern import six
from astropy.utils import lazyproperty

from ..helpers import sequentialfile as sf
from ..vlbi_base.base import (make_opener, VLBIFileBase, VLBIStreamBase,
                              VLBIStreamReaderBase, VLBIStreamWriterBase)
from .header import GUPPIHeader
from .payload import GUPPIPayload
from .frame import GUPPIFrame


__all__ = ['GUPPIFileReader', 'GUPPIFileWriter', 'GUPPIStreamBase',
           'GUPPIStreamReader', 'GUPPIStreamWriter', 'open']


class GUPPIFileReader(VLBIFileBase):
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
        return GUPPIFrame.fromfile(self.fh_raw, memmap=memmap)

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
            return (header.sample_rate /
                    (header.samples_per_frame - header.overlap))
        finally:
            self.seek(oldpos)


class GUPPIFileWriter(VLBIFileBase):
    """Simple writer/mapper for DADA files.

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

    def __init__(self, fh_raw, header0, squeeze=True, subset=()):

        # GUPPI sample positions are determined by "packets", the bytesize of
        # which are given in the header.
        self._packets_per_frame = (
            (header0.payload_nbytes -
             header0.overlap * header0.bits_per_complete_sample // 8) //
            header0['PKTSIZE'])

        # Set samples per frame to valid ones only.
        samples_per_frame = header0.samples_per_frame - header0.overlap

        super(GUPPIStreamBase, self).__init__(
            fh_raw=fh_raw, header0=header0, sample_rate=header0.sample_rate,
            samples_per_frame=samples_per_frame,
            unsliced_shape=header0.sample_shape, bps=header0.bps,
            complex_data=header0.complex_data, squeeze=squeeze, subset=subset,
            fill_value=0.)

    # Overriding so the docstring indicates the exclusion of the overlap.
    samples_per_frame = property(VLBIStreamBase.samples_per_frame.fget,
                                 VLBIStreamBase.samples_per_frame.fset,
                                 doc=("Number of complete samples per frame, "
                                      "excluding overlap."))


class GUPPIStreamReader(GUPPIStreamBase, VLBIStreamReaderBase):
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
        fh_raw = GUPPIFileReader(fh_raw)
        header0 = GUPPIHeader.fromfile(fh_raw)
        super(GUPPIStreamReader, self).__init__(fh_raw, header0,
                                                squeeze=squeeze, subset=subset)

    @lazyproperty
    def _last_header(self):
        """Header of the last file for this stream."""
        nframes, fframe = divmod(self.fh_raw.seek(0, 2),
                                 self.header0.frame_nbytes)
        # If there is a non-integer number of frames, assume it's the last
        # frame that's missing bytes, and go to the last full frame.
        if fframe:
            self.fh_raw.seek((nframes - 1) * self.header0.frame_nbytes)
        # Otherwise go to the last frame.
        else:
            self.fh_raw.seek(-self.header0.frame_nbytes, 2)
        last_frame = self.fh_raw.read_frame(memmap=True)
        return last_frame.header

    def _read_frame(self, index):
        self.fh_raw.seek(index * self.header0.frame_nbytes)
        frame = self.fh_raw.read_frame(memmap=True)
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
        Header for the first frame, holding time information, etc.
    squeeze : bool, optional
        If `True` (default), `write` accepts squeezed arrays as input,
        and adds any dimensions of length unity.
    """
    def __init__(self, fh_raw, header0, squeeze=True):
        assert header0.get('OVERLAP', 0) == 0
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


opener = make_opener('GUPPI', globals(), doc="""
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
    with any further keyword arguments.

--- Header keywords : (see :meth:`~baseband.dada.DADAHeader.fromvalues`)

time : `~astropy.time.Time`
    Start time of the file.  Must have an integer number of seconds.
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
    Number of channels (default: 1).  For GUPPI, complex data is only allowed
    when nchan > 1.
bps : int, optional
    Bits per elementary sample, i.e. per real or imaginary component for
    complex data (default: 8).

Returns
-------
Filehandle
    :class:`~baseband.guppi.base.GUPPIFileReader` or
    :class:`~baseband.guppi.base.GUPPIFileWriter` (binary), or
    :class:`~baseband.guppi.base.GUPPIStreamReader` or
    :class:`~baseband.guppi.base.GUPPIStreamWriter` (stream).
""")


# Need to wrap the opener to be able to deal with file lists or templates.
# TODO: move this up to the opener??
def open(name, mode='rs', **kwargs):
    frames_per_file = kwargs.pop('frames_per_file', 128)
    header0 = kwargs.get('header0', None)
    squeeze = kwargs.pop('squeeze', None)
    # If sequentialfile object, check that it's opened properly.
    if isinstance(name, sf.SequentialFileBase):
        assert (('r' in mode and name.mode == 'rb') or
                ('w' in mode and name.mode == 'w+b')), (
                    "open only accepts sequential files opened in 'rb' mode "
                    "for reading or 'w+b' mode for writing.")
    is_sequence = isinstance(name, (tuple, list))

    if 'b' not in mode:
        if header0 is None:
            if 'w' in mode:
                # For writing a header is required.
                header0 = GUPPIHeader.fromvalues(**kwargs)
                kwargs = {}
            else:
                header0 = {}

        if is_sequence:  #is_template or is_sequence:
            if 'r' in mode:
                name = sf.open(name, 'rb')
            else:
                name = sf.open(name, 'w+b', file_size=(
                    frames_per_file * header0.frame_nbytes))

        if header0 and 'w' in mode:
            kwargs['header0'] = header0

    if squeeze is not None:
        kwargs['squeeze'] = squeeze

    return opener(name, mode, **kwargs)


open.__doc__ = opener.__doc__ + """\n
Notes
-----
For streams, one can also pass in a list of files, or equivalently a
`~baseband.helpers.sequentialfile` object (opened in 'rb' mode for reading or
'w+b' for writing).
"""
