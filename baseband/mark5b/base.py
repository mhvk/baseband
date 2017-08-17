import numpy as np
from astropy import units as u

from ..vlbi_base.base import (VLBIFileBase, VLBIStreamReaderBase,
                              VLBIStreamWriterBase, make_opener)
from .header import Mark5BHeader
from .frame import Mark5BFrame


__all__ = ['Mark5BFileReader', 'Mark5BFileWriter', 'Mark5BStreamReader',
           'Mark5BStreamWriter', 'open']


class Mark5BFileReader(VLBIFileBase):
    """Simple reader for Mark 5B files.

    Adds ``read_frame`` and ``find_header`` methods to the VLBI file wrapper.
    """

    def read_frame(self, ref_mjd, nchan, bps=2):
        """Read a single frame (header plus payload).

        Parameters
        ----------
        ref_time : int
            Used to determine the thousands in the Mark 5B header time.
            Thus, should be within 500 days of the actual observing time.
        nchan : int
            Number of channels encoded in the payload.
        bps : int
            Bits per sample (default=2).

        Returns
        -------
        frame : `~baseband.mark5b.Mark5BFrame`
            With ``header`` and ``data`` properties that return the
            Mark5BHeader and data encoded in the frame, respectively.
        """
        return Mark5BFrame.fromfile(self.fh_raw, nchan=nchan, bps=bps,
                                    ref_mjd=ref_mjd)

    def find_header(self, template_header=None, kday=None, framesize=None,
                    maximum=None, forward=True):
        """Look for the first occurrence of a frame.

        Search is from the current position.  If given, a template_header
        is used to initialize the framesize, as well as kday in the header.
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


class Mark5BStreamReader(VLBIStreamReaderBase, Mark5BFileReader):
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
    ref_mjd : int, or `~astropy.time.Time` instance
        Reference MJD (rounded to thousands), to remove ambiguities in the
        time stamps.  By default, will be inferred from the file creation date.
    thread_ids: list of int, optional
        Specific threads to read.  By default, all threads are read.
    frames_per_second : int, optional
        Needed to calculate timestamps. If not given, will be inferred from
        ``sample_rate``, or by scanning the file.
    sample_rate : `~astropy.units.Quantity`, optional
        Rate at which each thread is sampled (bandwidth * 2; frequency units).
    """

    _frame_class = Mark5BFrame

    def __init__(self, fh_raw, nchan, bps=2, ref_mjd=None, thread_ids=None,
                 frames_per_second=None, sample_rate=None):
        # Pre-set fh_raw, so FileReader methods work
        # TODO: move this to StreamReaderBase?
        self.fh_raw = fh_raw
        self._frame = self.read_frame(ref_mjd=ref_mjd, nchan=nchan, bps=bps)
        self._frame_data = None
        header = self._frame.header
        super(Mark5BStreamReader, self).__init__(
            fh_raw, header0=header, nchan=nchan, bps=bps, complex_data=False,
            thread_ids=thread_ids,
            samples_per_frame=header.payloadsize * 8 // bps // nchan,
            frames_per_second=frames_per_second, sample_rate=sample_rate)

    @property
    def size(self):
        n_frames = int(round(
            (self.header1.time - self.header0.time).to(u.s).value *
            self.frames_per_second)) + 1
        return n_frames * self.samples_per_frame

    def read(self, count=None, fill_value=0., squeeze=True, out=None):
        """Read count samples.

        The range retrieved can span multiple frames.

        Parameters
        ----------
        count : int
            Number of samples to read.  If omitted or negative, the whole
            file is read.
        fill_value : float or complex
            Value to use for invalid or missing data.
        squeeze : bool
            If `True` (default), remove channel and thread dimensions if unity.
        out : `None` or array
            Array to store the data in. If given, count will be inferred,
            and squeeze is set to `False`.

        Returns
        -------
        out : array of float or complex
            Dimensions are (sample-time, vlbi-thread, channel).
        """
        if out is None:
            if count is None or count < 0:
                count = self.size - self.offset

            out = np.empty((self.nthread, count),
                           dtype=self._frame.dtype).T
        else:
            count = out.shape[0]
            squeeze = False

        offset0 = self.offset
        while count > 0:
            dt, frame_nr, sample_offset = self._frame_info()
            if(dt != self._frame.seconds - self.header0.seconds or
               frame_nr != self._frame['frame_nr']):
                # Read relevant frame, reusing data array from previous frame.
                self._read_frame()
                assert dt == (self._frame.seconds - self.header0.seconds)
                assert frame_nr == self._frame['frame_nr']

            data = self._frame.data
            if self.thread_ids:
                data = data[:, self.thread_ids]
            # Copy relevant data from frame into output.
            nsample = min(count, self.samples_per_frame - sample_offset)
            sample = self.offset - offset0
            out[sample:sample + nsample] = data[sample_offset:
                                                sample_offset + nsample]
            self.offset += nsample
            count -= nsample

        return out.squeeze() if squeeze else out

    def _read_frame(self):
        self.fh_raw.seek(self.offset // self.samples_per_frame *
                         self._frame.size)
        self._frame = self.read_frame(ref_mjd=self.header0.kday,
                                      nchan=self.nchan, bps=self.bps)
        # Convert payloads to data array.
        self._frame_data = self._frame.data


class Mark5BStreamWriter(VLBIStreamWriterBase, Mark5BFileWriter):
    """VLBI Mark 5B format writer.

    Parameters
    ----------
    raw : `~baseband.mark5b.base.Mark5BFileWriter` instance.
        Which will write filled sets of frames to storage.
    frames_per_second : int, optional
        Needed to calculate timestamps. If not given, inferred from
        ``sample_rate``.
    sample_rate : `~astropy.units.Quantity`, optional
        Rate at which each thread is sampled (bandwidth * 2; frequency units).
    nchan : int, optional
        Number of threads the VLBI data has (e.g., 2 for 2 polarisations).
        Default is 1.
    bps : int
        Bits per sample.  Default is 2.
    header : `~baseband.mark5b.Mark5BHeader`, optional
        Header for the first frame, holding time information, etc.
    **kwargs
        If no header is give, an attempt is made to construct the header from
        these.  For a standard header, the following suffices.

    --- Header keywords : (see :meth:`~baseband.mark5b.Mark5BHeader.fromvalues`)

    time : `~astropy.time.Time` instance
        Sets bcd-encoded unit day, hour, minute, second, and fraction, as
        well as the frame number.
    """

    _frame_class = Mark5BFrame

    def __init__(self, raw, frames_per_second=None, sample_rate=None,
                 nchan=1, bps=2, header=None, **kwargs):
        if header is None:
            header = Mark5BHeader.fromvalues(**kwargs)
        super(Mark5BStreamWriter, self).__init__(
            raw, header0=header, nchan=nchan, bps=bps, complex_data=False,
            thread_ids=None,
            samples_per_frame=header.payloadsize * 8 // bps // nchan,
            frames_per_second=frames_per_second, sample_rate=sample_rate)
        self._data = np.zeros((self.samples_per_frame, self.nchan), np.float32)
        self._valid = True

    def write(self, data, squeezed=True, invalid_data=False):
        """Write data, buffering by frames as needed."""
        if squeezed and data.ndim < 2:
            data = np.expand_dims(data, axis=1 if self.nthread == 1 else 0)

        if data.ndim != 2 or data.shape[1] != self.nthread:
            raise ValueError('cannot write an array with shape {0} to a '
                             'stream with {1} threads'
                             .format(data.shape, self.nthread))

        count = data.shape[0]
        sample = 0
        offset0 = self.offset
        frame = self._data
        while count > 0:
            dt, frame_nr, sample_offset = self._frame_info()
            if sample_offset == 0:
                # set up header for new frame.
                self._header = self.header0.copy()
                self._header.update(time=self.tell(unit='time'),
                                    frame_nr=frame_nr)

            if invalid_data:
                # Mark whole frame as invalid data.
                self._valid = False

            nsample = min(count, self.samples_per_frame - sample_offset)
            sample_end = sample_offset + nsample
            sample = self.offset - offset0
            frame[sample_offset:sample_end] = data[sample:sample + nsample]
            if sample_end == self.samples_per_frame:
                self.write_frame(self._data, self._header,
                                 bps=self.bps, valid=self._valid)
                self._valid = True

            self.offset += nsample
            count -= nsample


open = make_opener('Mark5B', globals(), doc="""
--- For reading a stream : (see `~baseband.mark5b.base.Mark5BStreamReader`)

nchan : int
    Number of threads/channels stored in the file.
bps : int, optional
    Bits per sample.  Default: 2.
ref_mjd : int, or `~astropy.time.Time` instance
    Reference MJD (rounded to thousands), to remove ambiguities in the
    time stamps.  By default, will be inferred from the file creation date.
thread_ids: list of int, optional
    Specific threads to read.  By default, all threads are read.
frames_per_second : int, optional
    Needed to calculate timestamps. If not given, will be inferred from
    ``sample_rate``, or by scanning the file.
sample_rate : `~astropy.units.Quantity`, optional
    Rate at which each thread is sampled (bandwidth * 2; frequency units).

--- For writing a stream : (see `~baseband.mark5b.base.Mark5BStreamWriter`)

frames_per_second : int, optional
    Needed to calculate timestamps. If not given, inferred from
    ``sample_rate``.
sample_rate : `~astropy.units.Quantity`, optional
    Rate at which each thread is sampled (bandwidth * 2; frequency units).
nchan : int, optional
    Number of threads the VLBI data has (e.g., 2 for 2 polarisations).
    Default is 1.
bps : int
    Bits per sample.  Default is 2.
header : :class:`~baseband.mark5b.Mark5BHeader`, optional
    Header for the first frame, holding time information, etc.
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
