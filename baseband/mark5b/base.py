# Licensed under the GPLv3 - see LICENSE
from __future__ import division, unicode_literals, print_function

import numpy as np
import astropy.units as u
from astropy.utils import lazyproperty

from ..vlbi_base.base import (VLBIFileBase, VLBIFileReaderBase, VLBIStreamBase,
                              VLBIStreamReaderBase, VLBIStreamWriterBase,
                              make_opener)
from .header import Mark5BHeader
from .payload import Mark5BPayload
from .frame import Mark5BFrame
from .file_info import Mark5BFileReaderInfo


__all__ = ['Mark5BFileReader', 'Mark5BFileWriter', 'Mark5BStreamReader',
           'Mark5BStreamWriter', 'open']


class Mark5BFileReader(VLBIFileReaderBase):
    """Simple reader for Mark 5B files.

    Wraps a binary filehandle, providing methods to help interpret the data,
    such as `read_frame` and `get_frame_rate`.

    Parameters
    ----------
    fh_raw : filehandle
        Filehandle of the raw binary data file.
    kday : int or None
        Explicit thousands of MJD of the observation time.  Can instead
        pass an approximate ``ref_time``.
    ref_time : `~astropy.time.Time` or None
        Reference time within 500 days of the observation time, used to
        infer the full MJD.  Used only if ``kday`` is not given.
    nchan : int, optional
        Number of channels.   Default: 1.
    bps : int, optional
        Bits per elementary sample.  Default: 2.
    """
    def __init__(self, fh_raw, kday=None, ref_time=None, nchan=None, bps=2):
        self.kday = kday
        self.ref_time = ref_time
        self.nchan = nchan
        self.bps = bps
        super(Mark5BFileReader, self).__init__(fh_raw)

    def __repr__(self):
        return ("{name}(fh_raw={s.fh_raw}, kday={s.kday}, "
                "ref_time={s.ref_time}, nchan={s.nchan}, bps={s.bps})"
                .format(name=self.__class__.__name__, s=self))

    info = Mark5BFileReaderInfo()

    def read_header(self):
        """Read a single header from the file.

        Returns
        -------
        header : `~baseband.mark5b.Mark5BHeader`
        """
        return Mark5BHeader.fromfile(self, kday=self.kday,
                                     ref_time=self.ref_time)

    def read_frame(self, verify=True):
        """Read a single frame (header plus payload).

        Returns
        -------
        frame : `~baseband.mark5b.Mark5BFrame`
            With ``header`` and ``data`` properties that return the
            `~baseband.mark5b.Mark5BHeader` and data encoded in the frame,
            respectively.
        verify : bool, optional
            Whether to do basic checks of frame integrity.  Default: `True`.
        """
        if self.nchan is None:
            raise TypeError("In order to read frames, the file handle should "
                            "be initialized with nchan set.")
        return Mark5BFrame.fromfile(self.fh_raw, kday=self.kday,
                                    ref_time=self.ref_time, nchan=self.nchan,
                                    bps=self.bps, verify=verify)

    def get_frame_rate(self):
        """Determine the number of frames per second.

        This method first tries to determine the frame rate by looking for
        the highest frame number in the first second of data.  If that fails,
        it uses the time difference between two consecutive frames. This can
        fail if the headers do not store fractional seconds, or if the data
        rate is above 512 Mbps.

        Returns
        -------
        frame_rate : `~astropy.units.Quantity`
            Frames per second.
        """
        try:
            return super(Mark5BFileReader, self).get_frame_rate()
        except Exception as exc:
            oldpos = self.tell()
            self.seek(0)
            try:
                header0 = self.read_header()
                self.seek(header0.payload_nbytes, 1)
                header1 = self.read_header()
                tdelta = header1.fraction - header0.fraction
                if tdelta == 0.:
                    exc.args += ("frame rate can also not be determined "
                                 "from the first two headers, as they have "
                                 "identical fractional seconds.",)
                return np.round(1 / tdelta) * u.Hz
            except Exception:
                pass
            finally:
                self.seek(oldpos)
            raise exc

    def find_header(self, forward=True, maximum=None):
        """Find the nearest header from the current position.

        If successful, the file pointer is left at the start of the header.

        Parameters
        ----------
        forward : bool, optional
            Seek forward if `True` (default), backward if `False`.
        maximum : int, optional
            Maximum number of bytes to search through.  Default: twice the
            frame size of 10016 bytes.

        Returns
        -------
        header : :class:`~baseband.mark5b.Mark5BHeader` or None
            Retrieved Mark 5B header, or `None` if nothing found.
        """
        frame_nbytes = 10016  # This is fixed for Mark 5B.
        if maximum is None:
            maximum = 2 * frame_nbytes
        # Loop over chunks to try to find the frame marker.
        file_pos = self.tell()
        # First check whether we are right at a frame marker (usually true).
        try:
            header = self.read_header()
            self.seek(-header.nbytes, 1)
            return header
        except AssertionError:
            pass

        self.seek(0, 2)
        nbytes = self.tell()
        if forward:
            iterate = range(file_pos, min(file_pos + maximum - 16,
                                          nbytes - frame_nbytes))
        else:
            iterate = range(min(file_pos, nbytes - frame_nbytes),
                            max(file_pos - maximum, -1), -1)

        for frame in iterate:
            try:
                self.seek(frame)
                header1 = self.read_header()
            except AssertionError:
                continue

            # Get header from a frame up and check it is consistent (we always
            # check up since this checks that the payload has the right length)
            next_frame = frame + frame_nbytes
            if next_frame > nbytes - 16:
                # If we're too far ahead for there to be another header,
                # at least the one below should be OK.
                next_frame = frame - frame_nbytes
                # Except if there is only one frame in the first place.
                if next_frame < 0:
                    self.seek(frame)
                    return header1

            self.seek(next_frame)
            try:
                header2 = self.read_header()
            except AssertionError:
                continue

            if(header2.jday == header1.jday and
               abs(header2.seconds - header1.seconds) <= 1 and
               abs(header2['frame_nr'] - header1['frame_nr']) <= 1):
                self.seek(frame)
                return header1

        # Didn't find any frame.
        self.seek(file_pos)
        return None


class Mark5BFileWriter(VLBIFileBase):
    """Simple writer for Mark 5B files.

    Adds `write_frame` method to the VLBI binary file wrapper.
    """
    def write_frame(self, data, header=None, bps=2, valid=True, **kwargs):
        """Write a single frame (header plus payload).

        Parameters
        ----------
        data : `~numpy.ndarray` or :`~baseband.mark5b.Mark5BFrame`
            If an array, ``header`` should be given, which will be used to
            get the information needed to encode the array, and to construct
            the Mark 5B frame.
        header : `~baseband.mark5b.Mark5BHeader`
            Can instead give keyword arguments to construct a header.  Ignored
            if ``data`` is a `~baseband.mark5b.Mark5BFrame` instance.
        bps : int, optional
            Bits per elementary sample, to use when encoding the payload.
            Ignored if ``data`` is a `~baseband.mark5b.Mark5BFrame` instance.
            Default: 2.
        valid : bool, optional
            Whether the data are valid; if `False`, a payload filled with an
            appropriate pattern will be crated.  Ignored if ``data`` is a
            `~baseband.mark5b.Mark5BFrame` instance.  Default: `True`.
        **kwargs
            If ``header`` is not given, these are used to initialize one.
        """
        if not isinstance(data, Mark5BFrame):
            data = Mark5BFrame.fromdata(data, header, bps=bps, valid=valid,
                                        **kwargs)
        return data.tofile(self.fh_raw)


class Mark5BStreamBase(VLBIStreamBase):
    """Base for Mark 5B streams."""

    def __init__(self, fh_raw, header0, sample_rate=None, nchan=1,
                 bps=2, squeeze=True, subset=(), fill_value=0., verify=True):
        super(Mark5BStreamBase, self).__init__(
            fh_raw, header0=header0, sample_rate=sample_rate,
            samples_per_frame=header0.payload_nbytes * 8 // bps // nchan,
            unsliced_shape=(nchan,), bps=bps, complex_data=False,
            squeeze=squeeze, subset=subset, fill_value=fill_value,
            verify=verify)
        self._frame_rate = int(round((self.sample_rate /
                                      self.samples_per_frame).to_value(u.Hz)))


class Mark5BStreamReader(Mark5BStreamBase, VLBIStreamReaderBase):
    """VLBI Mark 5B format reader.

    Allows access a Mark 5B file as a continues series of samples.

    Parameters
    ----------
    fh_raw : filehandle
        Filehandle of the raw Mark 5B stream.
    sample_rate : `~astropy.units.Quantity`, optional
        Number of complete samples per second, i.e. the rate at which each
        channel is sampled.  If `None` (default), will be inferred from
        scanning one second of the file or, failing that, using the time
        difference between two consecutive frames.
    kday : int or None
        Explicit thousands of MJD of the observation start time (eg. ``57000``
        for MJD 57999), used to infer the full MJD from the header's time
        information.  Can instead pass an approximate ``ref_time``.
    ref_time : `~astropy.time.Time` or None
        Reference time within 500 days of the observation start time, used
        to infer the full MJD.  Only used if ``kday`` is not given.
    nchan : int
        Number of channels.  Needs to be explicitly passed in.
    bps : int, optional
        Bits per elementary sample.  Default: 2.
    squeeze : bool, optional
        If `True` (default), remove any dimensions of length unity from
        decoded data.
    subset : indexing object, optional
        Specific channels of the complete sample to decode (after possible
        squeezing). If an empty tuple (default), all channels are read.
    fill_value : float or complex
        Value to use for invalid or missing data. Default: 0.
    verify : bool, optional
        Whether to do basic checks of frame integrity when reading.  The first
        frame of the stream is always checked.  Default: `True`.
    """

    _sample_shape_maker = Mark5BPayload._sample_shape_maker

    def __init__(self, fh_raw, sample_rate=None, kday=None, ref_time=None,
                 nchan=None, bps=2, squeeze=True, subset=(), fill_value=0.,
                 verify=True):

        if nchan is None:
            raise TypeError("Mark 5B stream reader requires nchan to be "
                            "explicity passed in.")

        if kday is None and ref_time is None:
            raise TypeError("Mark 5B stream reader requires either kday or "
                            "ref_time to be passed in.")

        fh_raw = Mark5BFileReader(fh_raw, nchan=nchan, bps=bps,
                                  ref_time=ref_time, kday=kday)
        header0 = fh_raw.find_header()
        super(Mark5BStreamReader, self).__init__(
            fh_raw, header0, sample_rate=sample_rate, nchan=nchan, bps=bps,
            squeeze=squeeze, subset=subset, fill_value=fill_value,
            verify=verify)
        # Use ref_time in preference to kday so we can handle files that
        # span a change in 1000s of MJD.
        self.fh_raw.kday = None
        self.fh_raw.ref_time = self.start_time

    @lazyproperty
    def _last_header(self):
        """Last header of the file."""
        last_header = super(Mark5BStreamReader, self)._last_header
        # Infer kday, assuming the end of the file is no more than
        # 500 days away from the start.
        last_header.infer_kday(self.start_time)
        return last_header

    def _read_frame(self, index):
        self.fh_raw.seek(index * self.header0.frame_nbytes)
        frame = self.fh_raw.read_frame(verify=self.verify)
        # Set decoded value for invalid data.
        frame.fill_value = self.fill_value
        # TODO: OK to ignore leap seconds? Not sure what writer does.
        assert (self._frame_rate *
                (frame.seconds - self.header0.seconds +
                 86400 * (frame.kday + frame.jday -
                          self.header0.kday - self.header0.jday)) +
                frame['frame_nr'] - self.header0['frame_nr']) == index
        return frame


class Mark5BStreamWriter(Mark5BStreamBase, VLBIStreamWriterBase):
    """VLBI Mark 5B format writer.

    Encodes and writes sequences of samples to file.

    Parameters
    ----------
    fh_raw : filehandle
        For writing filled sets of frames to storage.
    header0 : `~baseband.mark5b.Mark5BHeader`
        Header for the first frame, holding time information, etc.  Can instead
        give keyword arguments to construct a header (see ``**kwargs``).
    sample_rate : `~astropy.units.Quantity`
        Number of complete samples per second, i.e. the rate at which each
        channel is sampled.  Needed to calculate header timestamps.
    nchan : int, optional
        Number of channels.  Default: 1.
    bps : int, optional
        Bits per elementary sample.  Default: 2.
    squeeze : bool, optional
        If `True` (default), `write` accepts squeezed arrays as input, and
        adds any dimensions of length unity.
    **kwargs
        If no header is given, an attempt is made to construct one from these.
        For a standard header, the following suffices.

    --- Header kwargs : (see :meth:`~baseband.mark5b.Mark5BHeader.fromvalues`)

    time : `~astropy.time.Time`
        Start time of the file.  Sets bcd-encoded unit day, hour, minute,
        second, and fraction, as well as the frame number, in the header.
    """

    _sample_shape_maker = Mark5BPayload._sample_shape_maker

    def __init__(self, fh_raw, header0=None, sample_rate=None, nchan=1, bps=2,
                 squeeze=True, **kwargs):
        samples_per_frame = Mark5BHeader._payload_nbytes * 8 // bps // nchan
        if header0 is None:
            if 'time' in kwargs:
                kwargs['frame_rate'] = sample_rate / samples_per_frame
            header0 = Mark5BHeader.fromvalues(**kwargs)

        fh_raw = Mark5BFileWriter(fh_raw)
        super(Mark5BStreamWriter, self).__init__(
            fh_raw, header0, sample_rate=sample_rate, nchan=nchan,
            bps=bps, squeeze=squeeze)
        # Initial payload, reused for every frame.
        self._payload = Mark5BPayload(np.zeros((2500,), np.uint32),
                                      nchan=self._unsliced_shape.nchan,
                                      bps=self.bps)

    def _make_frame(self, index):
        # Set up header for new frame.
        header = self.header0.copy()
        # Update time and frame_nr in one go.
        frame_rate = self._frame_rate * u.Hz
        header.set_time(time=self.start_time + index / frame_rate,
                        frame_rate=frame_rate)
        # Recalculate CRC.
        header.update()
        # Reuse payload.
        return Mark5BFrame(header, self._payload, valid=True)


open = make_opener('Mark5B', globals(), doc="""
--- For reading a stream : (see `~baseband.mark5b.base.Mark5BStreamReader`)

sample_rate : `~astropy.units.Quantity`, optional
    Number of complete samples per second, i.e. the rate at which each channel
    is sampled.  If `None` (default), will be inferred from scanning one
    second of the file or, failing that, using the time difference between two
    consecutive frames.
kday : int or None
    Explicit thousands of MJD of the observation start time (eg. ``57000`` for
    MJD 57999), used to infer the full MJD from the header's time information.
    Can instead pass an approximate ``ref_time``.
ref_time : `~astropy.time.Time` or None
    Reference time within 500 days of the observation start time, used to infer
    the full MJD.  Only used if ``kday`` is not given.
nchan : int, optional
    Number of channels.  Default: 1.
bps : int, optional
    Bits per elementary sample.  Default: 2.
squeeze : bool, optional
    If `True` (default), remove any dimensions of length unity from
    decoded data.
subset : indexing object, optional
    Specific channels of the complete sample to decode (after possible
    squeezing). If an empty tuple (default), all channels are read.
fill_value : float or complex
    Value to use for invalid or missing data. Default: 0.
verify : bool, optional
    Whether to do basic checks of frame integrity when reading.  The first
    frame of the stream is always checked.  Default: `True`.

--- For writing a stream : (see `~baseband.mark5b.base.Mark5BStreamWriter`)

header0 : :class:`~baseband.mark5b.Mark5BHeader`
    Header for the first frame, holding time information, etc.  Can instead
    give keyword arguments to construct a header (see ``**kwargs``).
sample_rate : `~astropy.units.Quantity`
    Number of complete samples per second, i.e. the rate at which each
    channel is sampled.  Needed to calculate header timestamps.
nchan : int, optional
    Number of channels.  Default: 1.
bps : int, optional
    Bits per elementary sample.  Default: 2.
squeeze : bool, optional
    If `True` (default), writer accepts squeezed arrays as input,
    and adds channel and thread dimensions if they have length unity.
file_size : int or None, optional
    When writing to a sequence of files, the maximum size of one file in bytes.
    If `None` (default), the file size is unlimited, and only the first
    file will be written to.
**kwargs
    If no header is given, an attempt is made to construct one with any further
    keyword arguments.  See :class:`~baseband.mark5b.base.Mark5BStreamWriter`.

Returns
-------
Filehandle
    :class:`~baseband.mark5b.base.Mark5BFileReader` or
    :class:`~baseband.mark5b.base.Mark5BFileWriter` (binary), or
    :class:`~baseband.mark5b.base.Mark5BStreamReader` or
    :class:`~baseband.mark5b.base.Mark5BStreamWriter` (stream).

Notes
-----
One can also pass to ``name`` a list, tuple, or subclass of
`~baseband.helpers.sequentialfile.FileNameSequencer`.  For writing to multiple
files, the ``file_size`` keyword must be passed or only the first file will be
written to.  One may also pass in a `~baseband.helpers.sequentialfile` object
(opened in 'rb' mode for reading or 'w+b' for writing), though for typical use
cases it is practically identical to passing in a list or template.
""")
