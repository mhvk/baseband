# Licensed under the GPLv3 - see LICENSE.rst
from __future__ import division, unicode_literals, print_function

import numpy as np
import astropy.units as u
from astropy.utils import lazyproperty

from ..vlbi_base.base import (VLBIFileBase, VLBIStreamBase,
                              VLBIStreamReaderBase, VLBIStreamWriterBase,
                              make_opener)
from .header import Mark5BHeader
from .payload import Mark5BPayload
from .frame import Mark5BFrame


__all__ = ['Mark5BFileReader', 'Mark5BFileWriter', 'Mark5BStreamReader',
           'Mark5BStreamWriter', 'open']


class Mark5BFileReader(VLBIFileBase):
    """Simple reader for Mark 5B files.

    Adds ``read_frame`` and ``find_header`` methods to the VLBI file wrapper.
    """

    def read_frame(self, nchan, bps=2, kday=None, ref_time=None):
        """Read a single frame (header plus payload).

        Parameters
        ----------
        nchan : int
            Number of channels encoded in the payload.
        bps : int
            Bits per sample (default=2).
        kday : int, or None, optional
            Explicit thousands of MJD of the observation time.  Can instead
            pass an approximate `ref_time`.
        ref_time : `~astropy.time.Time`, or None, optional
            Reference time within 500 days of the observation time, used to
            infer the full MJD.  Used only if `kday` is ``None``.

        Returns
        -------
        frame : `~baseband.mark5b.Mark5BFrame`
            With ``header`` and ``data`` properties that return the
            Mark5BHeader and data encoded in the frame, respectively.
        """
        return Mark5BFrame.fromfile(self.fh_raw, nchan, bps=bps,
                                    kday=kday, ref_time=ref_time)

    def find_header(self, template_header=None, framesize=None, kday=None,
                    maximum=None, forward=True):
        """Look for the first occurrence of a frame.

        Search is from the current position.  If given, a template_header
        is used to initialize the framesize, as well as kday in the header.

        Parameters
        ----------
        template_header : :class:`~baseband.mark5b.Mark5BHeader`, optional
            Template Mark 5B header, from which `kday` and `framesize`
            are extracted.
        framesize : int, optional
            Size of a frame, in bytes.  Required if `template_header` is
            ``None``.
        kday : int, optional
            Explicit thousands of MJD of the observation time, used to infer
            the full MJD.  If `template_header` and `kday` are both ``None``,
            any header returned will have its `kday` set to ``None``.
        maximum : int, optional
            Maximum number of bytes to search through.  Default is twice the
            framesize.
        forward : bool, optional
            Seek forward if ``True`` (default), backward if ``False``.

        Returns
        -------
        header : :class:`~baseband.mark5b.Mark5BHeader`, or None
            Retrieved Mark 5B header, or ``None`` if nothing found.
        """
        fh = self.fh_raw
        if template_header:
            kday = template_header.kday
            framesize = template_header.framesize
        if maximum is None:
            maximum = 2 * framesize
        # Loop over chunks to try to find the frame marker.
        file_pos = fh.tell()
        # First check whether we are right at a frame marker (usually true).
        try:
            header = Mark5BHeader.fromfile(fh, kday=kday, verify=True)
            fh.seek(-header.size, 1)
            return header
        except AssertionError:
            pass

        fh.seek(0, 2)
        size = fh.tell()
        if forward:
            iterate = range(file_pos, min(file_pos + maximum - 16,
                                          size - framesize))
        else:
            iterate = range(min(file_pos, size - framesize),
                            max(file_pos - maximum, -1), -1)

        for frame in iterate:
            try:
                fh.seek(frame)
                header1 = Mark5BHeader.fromfile(fh, kday=kday, verify=True)
            except AssertionError:
                continue

            # get header from a frame up and check it is consistent (we always
            # check up since this checks that the payload has the right length)
            next_frame = frame + framesize
            if next_frame > size - 16:
                # if we're too far ahead for there to be another header,
                # at least the one below should be OK.
                next_frame = frame - framesize
                # except if there is only one frame in the first place.
                if next_frame < 0:
                    fh.seek(frame)
                    return header1

            fh.seek(next_frame)
            try:
                header2 = Mark5BHeader.fromfile(fh, kday=kday, verify=True)
            except AssertionError:
                continue

            if(header2.jday == header1.jday and
               abs(header2.seconds - header1.seconds) <= 1 and
               abs(header2['frame_nr'] - header1['frame_nr']) <= 1):
                fh.seek(frame)
                return header1

        # Didn't find any frame.
        fh.seek(file_pos)
        return None


class Mark5BFileWriter(VLBIFileBase):
    """Simple writer for Mark 5B files.

    Adds ``write_frame`` method to the VLBI binary file wrapper.
    """

    def write_frame(self, data, header=None, bps=2, valid=True, **kwargs):
        """Write a single frame (header plus payload).

        Parameters
        ----------
        data : array or :`~baseband.mark5b.Mark5BFrame`
            If an array, a `header` should be given, which will be used to
            get the information needed to encode the array, and to construct
            the Mark 5B frame.
        header : `~baseband.mark5b.Mark5BHeader`, optional
            Ignored if `data` is a Mark5B frame.
        bps : int, optional
            The number of bits per sample to be used to encode the payload.
            Ignored if `data` is a Mark5B frame.  Default: 2.
        valid : bool, optional
            Whether the data is valid; if `False`, a payload filled with an
            appropriate pattern will be crated.
            Ignored if `data` is a Mark5B frame.  Default: `True`
        **kwargs
            If no `header` is given, an attempt is made to initialize one
            using keywords arguments.
        """
        if not isinstance(data, Mark5BFrame):
            data = Mark5BFrame.fromdata(data, header, bps=bps, valid=valid,
                                        **kwargs)
        return data.tofile(self.fh_raw)


class Mark5BStreamBase(VLBIStreamBase):
    """Base for Mark 5B streams."""
    def __init__(self, fh_raw, header, nchan, bps=2, subset=None,
                 sample_rate=None, fill_value=0., squeeze=True):
        super(Mark5BStreamBase, self).__init__(
            fh_raw, header0=header, bps=bps, complex_data=False, subset=subset,
            unsliced_shape=(nchan,),
            samples_per_frame=header.payloadsize * 8 // bps // nchan,
            sample_rate=sample_rate, fill_value=fill_value, squeeze=squeeze)
        self._framerate = int(round((self.sample_rate /
                                     self.samples_per_frame).to_value(u.Hz)))


class Mark5BStreamReader(Mark5BStreamBase, VLBIStreamReaderBase,
                         Mark5BFileReader):
    """VLBI Mark 5B format reader.

    This wrapper is allows one to access a Mark 5B file as a continues series
    of samples.  Note that possible gaps in the data stream are not filled in.

    Parameters
    ----------
    fh_raw : `~baseband.mark5b.base.Mark5BFileReader` instance
        file handle of the raw Mark 5B stream
    nchan : int
        Number of threads/channels stored in the file.
    bps : int, optional
        Bits per sample.  Default: 2.
    kday : int, or None, optional
        Explicit thousands of MJD of the observation start time (eg. ``57000``
        for MJD 57999), used to infer the full MJD from the header's time
        information.  Can instead pass an approximate `ref_time`.
    ref_time : `~astropy.time.Time`, or None, optional
        Reference time within 500 days of the observation start time, used
        to infer the full MJD.  Only used if `kday` is ``None``.
    subset : indexing object, optional
        Specific components (ie. channels) of the complete sample to decode.
        By default, all channels are read.
    sample_rate : `~astropy.units.Quantity`, optional
        Number of complete samples per second (ie. the rate at which each
        channel is sampled).  If not given, will be inferred from scanning
        one second of the file.
    fill_value : float or complex
        Value to use for invalid or missing data. Default: 0.
    squeeze : bool, optional
        If `True` (default), remove any dimensions of length unity from
        decoded data.
    """

    _sample_shape_maker = Mark5BPayload._sample_shape_maker

    def __init__(self, fh_raw, nchan, bps=2, kday=None, ref_time=None,
                 subset=None, sample_rate=None, fill_value=0., squeeze=True):

        if kday is None and ref_time is None:
            raise ValueError("Mark5B stream reader requires kday or ref_time. "
                             "Please pass either explicitly.")

        header = Mark5BHeader.fromfile(fh_raw, ref_time=ref_time, kday=kday)
        # Go back to start of frame (for possible sample rate detection).
        fh_raw.seek(0)
        super(Mark5BStreamReader, self).__init__(
            fh_raw, header=header, nchan=nchan, bps=bps, subset=subset,
            sample_rate=sample_rate, fill_value=fill_value, squeeze=squeeze)

    @lazyproperty
    def _last_header(self):
        """Last header of the file."""
        last_header = super(Mark5BStreamReader, self)._last_header
        # Infer kday, assuming the end of the file is no more than
        # 500 days away from the start.
        last_header.infer_kday(self.start_time)
        return last_header

    def _read_frame(self, index):
        self.fh_raw.seek(index * self.header0.framesize)
        frame = self.read_frame(nchan=self._unsliced_shape.nchan,
                                bps=self.bps, ref_time=self.start_time)
        # Set decoded value for invalid data.
        frame.invalid_data_value = self.fill_value
        # TODO: OK to ignore leap seconds? Not sure what writer does.
        assert (self._framerate *
                (frame.seconds - self.header0.seconds +
                 86400 * (frame.kday + frame.jday -
                          self.header0.kday - self.header0.jday)) +
                frame['frame_nr'] - self.header0['frame_nr']) == index
        return frame


class Mark5BStreamWriter(Mark5BStreamBase, VLBIStreamWriterBase,
                         Mark5BFileWriter):
    """VLBI Mark 5B format writer.

    Parameters
    ----------
    raw : `~baseband.mark5b.base.Mark5BFileWriter` instance.
        Which will write filled sets of frames to storage.
    sample_rate : `~astropy.units.Quantity`
        Number of complete samples per second (ie. the rate at which each
        channel is sampled), needed to calculate header timestamps.
    nchan : int, optional
        Number of threads the VLBI data has (e.g., 2 for 2 polarisations).
        Default is 1.
    bps : int
        Bits per sample.  Default is 2.
    header : `~baseband.mark5b.Mark5BHeader`, optional
        Header for the first frame, holding time information, etc.
    squeeze : bool, optional
        If `True` (default), ``write`` accepts squeezed arrays as input,
        and adds channel and thread dimensions if they have length unity.
    **kwargs
        If no header is give, an attempt is made to construct the header from
        these.  For a standard header, the following suffices.

    --- Header kwargs : (see :meth:`~baseband.mark5b.Mark5BHeader.fromvalues`)

    time : `~astropy.time.Time` instance
        Sets bcd-encoded unit day, hour, minute, second, and fraction, as
        well as the frame number.
    """

    _sample_shape_maker = Mark5BPayload._sample_shape_maker

    def __init__(self, raw, sample_rate, nchan=1, bps=2, header=None,
                 squeeze=True, **kwargs):
        samples_per_frame = Mark5BHeader._payloadsize * 8 // bps // nchan
        if header is None:
            if 'time' in kwargs:
                kwargs['framerate'] = sample_rate / samples_per_frame
            header = Mark5BHeader.fromvalues(**kwargs)
        super(Mark5BStreamWriter, self).__init__(
            raw, header=header, nchan=nchan, bps=bps,
            sample_rate=sample_rate, squeeze=squeeze)
        # Initial payload, reused for every frame.
        self._payload = Mark5BPayload(np.zeros((2500,), np.uint32),
                                      nchan=self._unsliced_shape.nchan,
                                      bps=self.bps)

    def _make_frame(self, index):
        # set up header for new frame.
        header = self.header0.copy()
        # Update time and frame_nr in one go.
        # (Note: could also pass on frame rate instead of explicit frame)
        header.set_time(time=self.start_time + index / self._framerate * u.s,
                        frame_nr=((self.header0['frame_nr'] + index) %
                                  self._framerate))
        # Recalculate CRC.
        header.update()
        # Reuse payload.
        return Mark5BFrame(header, self._payload, valid=True)


open = make_opener('Mark5B', globals(), doc="""
--- For reading a stream : (see `~baseband.mark5b.base.Mark5BStreamReader`)

nchan : int
    Number of threads/channels stored in the file.
bps : int, optional
    Bits per sample.  Default: 2.
kday : int, or None, optional
    Explicit thousands of MJD of the observation start time (eg. ``57000`` for
    MJD 57999), used to infer the full MJD from the header's time information.
    Can instead pass an approximate `ref_time`.
ref_time : `~astropy.time.Time`, or None, optional
    Reference time within 500 days of the observation start time, used to infer
    the full MJD.  Only used if `kday` is ``None``.
subset : slice or other indexing object, optional
    Specific components (ie. channels) of the complete sample to decode.  By
    default, all channels are read.
sample_rate : `~astropy.units.Quantity`, optional
    Number of complete samples per second (ie. the rate at which each channel
    is sampled).  If not given, will be inferred from scanning one second of
    the file.
fill_value : float or complex
    Value to use for invalid or missing data. Default: 0.
squeeze : bool, optional
    If `True` (default), remove any dimensions of length unity from
    decoded data.

--- For writing a stream : (see `~baseband.mark5b.base.Mark5BStreamWriter`)

sample_rate : `~astropy.units.Quantity`
    Number of complete samples per second (ie. the rate at which each
    channel is sampled), needed to calculate header timestamps.
nchan : int, optional
    Number of threads the VLBI data has (e.g., 2 for 2 polarisations).
    Default is 1.
bps : int
    Bits per sample.  Default is 2.
header : :class:`~baseband.mark5b.Mark5BHeader`, optional
    Header for the first frame, holding time information, etc.
squeeze : bool, optional
    If `True` (default), ``write`` accepts squeezed arrays as input,
    and adds channel and thread dimensions if they have length unity.
**kwargs
    If the header is not given, an attempt will be made to construct one
    with any further keyword arguments.  See
    :class:`~baseband.mark5b.base.Mark5BStreamWriter`.

Returns
-------
Filehandle
    :class:`~baseband.mark5b.base.Mark5BFileReader` or
    :class:`~baseband.mark5b.base.Mark5BFileWriter` instance (binary), or
    :class:`~baseband.mark5b.base.Mark5BStreamReader` or
    :class:`~baseband.mark5b.base.Mark5BStreamWriter` instance (stream).
""")
