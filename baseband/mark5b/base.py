import os
import io

import numpy as np
from astropy import units as u
from astropy.time import Time

from ..vlbi_base.base import VLBIStreamReaderBase, VLBIStreamWriterBase
from .header import Mark5BHeader
from .frame import Mark5BFrame


__all__ = ['Mark5BFileReader', 'Mark5BFileWriter', 'Mark5BStreamReader',
           'Mark5BStreamWriter', 'open']


class Mark5BFileReader(io.BufferedReader):
    """Simple reader for Mark 5B files.

    Adds ``read_frame`` method to the basic binary file reader
    :class:`~io.BufferedReader`.
    """

    def read_frame(self, nchan, bps=2, ref_mjd=None):
        """Read a single frame (header plus payload).

        Parameters
        ----------
        nchan : int
            Number of channels encoded in the payload
        bps : int
            Bits per sample (default=2).
        ref_time : MJD (or astropy.time.Time instance)
            Used to determine the thousands in the Mark 5B header time.
            Thus, should be within 500 days of the actual observing time.
            By default, the file creation time is used.

        Returns
        -------
        frame : Mark5BFrame
            With ''.header'' and ''.data'' properties that return the
            Mark5BHeader and data encoded in the frame, respectively.
        """
        if ref_mjd is None:
            if not hasattr(self, '_creation_time_mjd'):
                self._creation_time_mjd = Time(os.path.getctime(self.name),
                                               format='unix').mjd
            ref_mjd = self._creation_time_mjd
        if isinstance(ref_mjd, Time):
            ref_mjd = ref_mjd.mjd
        return Mark5BFrame.fromfile(self, nchan=nchan, bps=bps,
                                    ref_mjd=ref_mjd)

    def find_frame(self, kday=None, template_header=None, framesize=None,
                   maximum=None, forward=True):
        """Look for the first occurrence of a frame.

        Search is from the current position.  If given, a template_header
        is used to initialize the framesize, as well as kday in the header.
        """
        if template_header:
            kday = template_header.kday
            framesize = template_header.framesize
        if maximum is None:
            maximum = 2 * framesize
        # Loop over chunks to try to find the frame marker.
        file_pos = self.tell()
        # First check whether we are right at a frame marker (usually true).
        try:
            header = Mark5BHeader.fromfile(self, kday=kday, verify=True)
            self.seek(-header.size, 1)
            return header
        except:
            pass

        if forward:
            iterate = range(file_pos, file_pos + maximum)
        else:
            iterate = range(file_pos, file_pos - maximum, -1)
        for frame in iterate:
            try:
                self.seek(frame)
                header1 = Mark5BHeader.fromfile(self, kday=kday, verify=True)
            except AssertionError:
                continue
            except:
                break

            # get header from a frame further up or down and check those are
            # consistent.
            try:
                self.seek(frame + (framesize if forward else -framesize))
            except:  # we're a really short file; assume it is fine.
                self.seek(frame)
                return header1

            try:
                header2 = Mark5BHeader.fromfile(self, kday=kday, verify=True)
            except AssertionError:
                continue
            except:
                break

            if(header2.jday == header1.jday and
               abs(header2.seconds - header1.seconds) <= 1 and
               abs(header2['frame_nr'] - header1['frame_nr']) <= 1):
                self.seek(frame)
                return header1

        # Didn't find any frame.
        self.seek(file_pos)
        return None


class Mark5BFileWriter(io.BufferedWriter):
    """Simple writer for Mark 5B files.

    Adds ``write_frame`` method to the basic binary file writer
    :class:`~io.BufferedWriter`.
    """
    def write_frame(self, data, header=None, bps=2, **kwargs):
        """Write a single frame (header plus payload).

        Parameters
        ----------
        data : array or :class:`~baseband.mark5b.Mark5BFrame`
            If an array, a `header` should be given, which will be used to
            get the information needed to encode the array, and to construct
            the Mark 5B frame.
        header : :class:`~baseband.mark5b.Mark5BHeader`, optional
            Ignored if `data` is a Mark5B frame.
        bps : int, optional
            Ignored if `data` is a Mark5B frame.  Default: 2.
        **kwargs
            If no `header` is given, an attempt is made to initialize one
            using keywords arguments.
        """
        if not isinstance(data, Mark5BFrame):
            if header is None:
                header = Mark5BHeader.fromvalues(**kwargs)
            data = Mark5BFrame.fromdata(data, header, bps=bps)
        return data.tofile(self)


class Mark5BStreamReader(VLBIStreamReaderBase):
    """VLBI Mark 5B format reader.

    This wrapper is allows one to access a Mark 5B file as a continues series
    of samples.  Note that possible gaps in the data stream are not filled in.

    Parameters
    ----------
    name : str
        file name
    nchan : int
        Number of threads stored in the file.
    bps : int, optional
        Bits per sample.  Default: 2.
    thread_ids: list of int, optional
        Specific threads to read.  By default, all threads are read.
    sample_rate : :class`~astropy.units.Quantity`, optional
        Rate at which each thread is sampled (bandwidth * 2; frequency units).
        If not given, it will be determined from the frame rate.
    """

    _frame_class = Mark5BFrame

    def __init__(self, raw, nchan, bps=2, ref_mjd=None, thread_ids=None,
                 sample_rate=None):
        if not hasattr(raw, 'read'):
            raw = io.open(raw, mode='rb')
        if not isinstance(raw, Mark5BFileReader):
            raw = Mark5BFileReader(raw)
        self._frame = raw.read_frame(nchan, bps, ref_mjd)
        self._frame_data = None
        super(Mark5BStreamReader, self).__init__(
            raw, header0=self._frame.header, nchan=nchan, bps=bps,
            thread_ids=thread_ids, sample_rate=sample_rate)

    @property
    def size(self):
        n_frames = round(
            (self.header1.time - self.header0.time).to(u.s).value *
            self.frames_per_second) + 1
        return n_frames * self.samples_per_frame

    def seek(self, offset, from_what=0):
        """Like normal seek, but with the offset in samples."""
        if from_what == 0:
            self.offset = offset
        elif from_what == 1:
            self.offset += offset
        elif from_what == 2:
            self.offset = self.size + offset
        return self.offset

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
            dt, frame_nr, sample_offset = self.tell(unit='frame_info')
            if(dt != self._frame.seconds - self.header0.seconds or
               frame_nr != self._frame['frame_nr']):
                # Read relevant frame, reusing data array from previous frame.
                self._read_frame(out=self._frame_data)
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

    def _read_frame(self, out=None):
        self.fh_raw.seek(self.offset // self.samples_per_frame *
                         self._frame.size)
        self._frame = self.fh_raw.read_frame(nchan=self.nchan, bps=self.bps)
        # Convert payloads to data array.
        self._frame_data = self._frame.todata(data=out)
        return self._frame_data


class Mark5BStreamWriter(VLBIStreamWriterBase):
    """VLBI Mark 5B format writer.

    Parameters
    ----------
    raw : filehandle, or name.
        Should be a :class:`Mark5BFileWriter` or :class:`~io.BufferedWriter`
        instance. If a name, will get opened for writing binary data.
    nthread : int, optional
        Number of threads the VLBI data has (e.g., 2 for 2 polarisations).
        Default is 1.
    header : :class:`~baseband.mark5b.Mark5BHeader`, optional
        Header for the first frame, holding time information, etc.

    If no header is give, an attempt is made to construct the header from the
    remaining keyword arguments.  For a standard header, this would include:

    time : `~astropy.time.Time` instance
        Or 'ref_epoch' + 'seconds'
    nchan : number of FFT channels within stream (default 1).
        Note: that different # of channels per thread is not supported.
    frame_length : number of long words for header plus payload
        For some edv, this is fixed (e.g., 629 for edv=3).
    complex_data : whether data is complex
    bps : bits per sample
        Or 'bits_per_sample', which is bps-1.
    station_id : 2 characters
        Or unsigned 2-byte integer.
    edv : 1, 3, or 4

    For edv = 1, 3, or 4, in addition, a required keyword is

    bandwidth : Quantity in Hz
        Or 'sampling_unit' + 'sample_rate'.

    For other edv, one requires

    framerate : number of frames per second.
    """

    _frame_class = Mark5BFrame

    def __init__(self, raw, nthread=1, header=None, **kwargs):
        if isinstance(raw, io.BufferedWriter):
            if not isinstance(raw, Mark5BFileWriter):
                raw = Mark5BFileWriter(raw)
        else:
            raw = Mark5BFileWriter(io.open(raw, mode='wb'))
        if header is None:
            header = Mark5BHeader.fromvalues(**kwargs)
        super(Mark5BStreamWriter, self).__init__(raw, header, range(nthread))
        self._data = np.zeros(
            (self.nthread, self.samples_per_frame, self.nchan),
            np.complex64 if header['complex_data'] else np.float32)

    def write(self, data, squeezed=True, invalid_data=False):
        """Write data, buffering by frames as needed."""
        if squeezed and data.ndim < 3:
            if self.nthread == 1:
                data = np.expand_dims(data, axis=1)
            if self.nchan == 1:
                data = np.expand_dims(data, axis=-1)

        assert data.shape[1] == self.nthread
        assert data.shape[2] == self.nchan

        count = data.shape[0]
        sample = 0
        offset0 = self.offset
        frame = self._data.transpose(1, 0, 2)
        while count > 0:
            dt, frame_nr, sample_offset = self.tell(unit='frame_info')
            if sample_offset == 0:
                # set up header for new frame.
                self._header = self.header0.copy()
                self._header['seconds'] = self.header0['seconds'] + dt
                self._header['frame_nr'] = frame_nr

            if invalid_data:
                # Mark whole frame as invalid data.
                self._header['invalid_data'] = invalid_data

            nsample = min(count, self.samples_per_frame - sample_offset)
            sample_end = sample_offset + nsample
            sample = self.offset - offset0
            frame[sample_offset:sample_end] = data[sample:sample + nsample]
            if sample_end == self.samples_per_frame:
                self.fh_raw.write_frame(self._data, self._header)

            self.offset += nsample
            count -= nsample


def open(name, mode='rs', **kwargs):
    """Open VLBI Mark 5B format file for reading or writing.

    Opened as a binary file, one gets a standard file handle, but with
    methods to read/write a frame.  Opened as a stream, the file handler
    is wrapped, allowing access to it as a series of samples.

    Parameters
    ----------
    name : str
        File name
    mode : {'rb', 'wb', 'rs', or 'ws'}, optional
        Whether to open for reading or writing, and as a regular binary file
        or as a stream (default is reading a stream).
    **kwargs
        Additional arguments when opening the file as a stream

    --- For reading a stream : (see :class:`Mark5BStreamReader`)

    nchan : int
        Number of threads stored in the file.
    bps : int, optional
        Bits per sample.  Default: 2.
    thread_ids: list of int, optional
        Specific threads to read.  By default, all threads are read.
    sample_rate : :class`~astropy.units.Quantity`, optional
        Rate at which each thread is sampled (bandwidth * 2; frequency units).
        If not given, it will be determined from the frame rate.

    --- For writing a stream : (see :class:`Mark5BStreamWriter`)

    nthread : int, optional
        Number of threads the VLBI data has (e.g., 2 for 2 polarisations).
        Default is 1.
    header : :class:`~baseband.mark5b.Mark5BHeader`, optional
        Header for the first frame, holding time information, etc.
    **kwargs
        If the header is not given, an attempt will be made to construct one
        with any further keyword arguments.  See :class:`Mark5BStreamWriter`.

    Returns
    -------
    Filehandle
        :class:`Mark5BFileReader` or :class:`Mark5BFileWriter` instance
        (binary), or :class:`Mark5BStreamReader` or
        :class:`Mark5BStreamWriter` instance (stream).
    """
    if 'w' in mode:
        if not hasattr(name, 'write'):
            name = io.open(name, 'wb')
        fh = Mark5BFileWriter(name)
        return fh if 'b' in mode else Mark5BStreamWriter(fh, **kwargs)
    elif 'r' in mode:
        if not hasattr(name, 'read'):
            name = io.open(name, 'rb')
        fh = Mark5BFileReader(name)
        return fh if 'b' in mode else Mark5BStreamReader(fh, **kwargs)
    else:
        raise ValueError("Only support opening Mark 5B file for reading "
                         "or writing (mode='r' or 'w').")
