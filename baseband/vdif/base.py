import io
import warnings

import numpy as np
import astropy.units as u
from astropy.utils import lazyproperty

from ..vlbi_base import VLBIStreamBase
from .header import VDIFHeader
from .frame import VDIFFrame, VDIFFrameSet


u_sample = u.def_unit('sample')

# Check code on 2015-MAY-30
# 00000000  77 2c db 00 00 00 00 1c  75 02 00 20 fc ff 01 04  # header 0 - 3
# 00000010  10 00 80 03 ed fe ab ac  00 00 40 33 83 15 03 f2  # header 4 - 7
# 00000020  2a 0a 7c 43 8b 69 9d 59  cb 99 6d 9a 99 96 5d 67  # data 0 - 3
# NOTE: thread_id = 1
# 2a = 00 10 10 10 = (lsb first) 1,  1,  1, -3
# 0a = 00 00 10 10 =             1,  1, -3, -3
# 7c = 01 11 11 00 =            -3,  3,  3, -1
# m5d evn/Fd/GP052D_FD_No0006.m5a VDIF_5000-512-1-2 100
# Mark5 stream: 0x16cd140
#   stream = File-1/1=evn/Fd/GP052D_FD_No0006.m5a
#   format = VDIF_5000-512-1-2 = 3
#   start mjd/sec = 56824 21367.000000000
#   frame duration = 78125.00 ns
#   framenum = 0
#   sample rate = 256000000 Hz
#   offset = 0
#   framebytes = 5032 bytes
#   datasize = 5000 bytes
#   sample granularity = 4
#   frame granularity = 1
#   gframens = 78125
#   payload offset = 32
#   read position = 0
#   data window size = 1048576 bytes
#  1  1  1 -3  1  1 -3 -3 -3  3  3 -1  -> OK
# fh = vdif.open('evn/Fd/GP052D_FD_No0006.m5a', 'rb')
# fs = fh.read_frameset()
# fs.data.astype(int)[1, :12, 0]  # thread id = 1!!
# -> array([ 1,  1,  1, -3,  1,  1, -3, -3, -3,  3,  3, -1])  -> OK
# Also, next frame (thread #3)
# m5d evn/Fd/GP052D_FD_No0006.m5a VDIF_5000-512-1-2 12 5032
# -1  1 -1  1 -3 -1  3 -1  3 -3  1  3
# fs.data.astype(int)[3, :12, 0]
# -> array([-1,  1, -1,  1, -3, -1,  3, -1,  3, -3,  1,  3])
# And first thread #0
# m5d evn/Fd/GP052D_FD_No0006.m5a VDIF_5000-512-1-2 12 20128
# -1 -1  3 -1  1 -1  3 -1  1  3 -1  1
# fs.data.astype(int)[0, :12, 0]
# -> array([-1, -1,  3, -1,  1, -1,  3, -1,  1,  3, -1,  1])
# sanity check that we can read 12 samples with stream reader
# fh.close()
# fh = vdif.open('evn/Fd/GP052D_FD_No0006.m5a', 'rs')
# fh.read(12).astype(int)[:, 0]
# -> array([-1, -1,  3, -1,  1, -1,  3, -1,  1,  3, -1,  1])


class VDIFFileReader(io.BufferedReader):
    """Simple reader for VDIF files.

    Adds ``read_frame`` and ``read_frameset`` methods to the basic binary
    file reader ``io.BufferedReader``.
    """
    def read_frame(self):
        """Read a single frame (header plus payload).

        Returns
        -------
        frame : VDIFFrame
            With ''.header'' and ''.data'' properties that return the
            VDIFHeader and data encoded in the frame, respectively.
        """
        return VDIFFrame.fromfile(self)

    def read_frameset(self, thread_ids=None, sort=True, edv=None, verify=True):
        """Read a single frame (header plus payload).

        Parameters
        ----------
        thread_ids : list or None
            The thread ids that should be read.  If `None`, read all threads.
        sort : bool
            Whether to sort the frames by thread_id.  Default: True.
        edv : int or None
            The expected extended data version for the VDIF Header.  If not
            given, use that of the first frame.  (Passing it in slightly
            improves file integrity checking.)
        verify : bool
            Whether to do (light) sanity checks on the header. Default: True.

        Returns
        -------
        frameset : VDIFFrameSet
            With ''.headers'' and ''.data'' properties that return list of
            VDIFHeaders and data encoded in the frame, respectively.
        """
        return VDIFFrameSet.fromfile(self, thread_ids, sort=sort, edv=edv,
                                     verify=verify)


class VDIFFileWriter(io.BufferedWriter):
    """Simple writer for VDIF files.

    Adds ``write_frame`` and ``write_frameset`` methods to the basic binary
    file writer ``io.BufferedWriter``.
    """
    def write_frame(self, data, header=None, **kwargs):
        """Write a single frame (header plus payload).

        Parameters
        ----------
        data : array or VDIFFrame
            If an array, a header should be given, which will be used to
            get the information needed to encode the array, and to construct
            the VDIF frame.
        header : VDIFHeader or dict
            Ignored if payload is a VDIFFrame instance.  If None, an attempt is
            made to initiate a header with **kwargs.
        """
        if not isinstance(data, VDIFFrame):
            data = VDIFFrame.fromdata(data, header, **kwargs)
        return data.tofile(self)

    def write_frameset(self, data, header=None, **kwargs):
        """Write a frame set (headers plus payloads).

        Parameters
        ----------
        data : array or VDIFFrameSet
            If an array, a header should be given, which will be used to
            get the information needed to encode the array, and to construct
            the VDIF frame.
        header : list of VDIFHeader, VDIFHeader, or dict.
            Ignored if payload is a VDIFFrameSet instance.  If a list, should
            have a length matching the number of threads in ``data``; if a
            single VDIFHeader, thread_ids corresponding to the number of
            threads are generated automatically; if None, an attempt is made
            to initiate a header using **kwargs.
        """
        if not isinstance(data, VDIFFrameSet):
            data = VDIFFrameSet.fromdata(data, header, **kwargs)
        return data.tofile(self)


class VDIFStreamBase(VLBIStreamBase):
    """VDIF file wrapper, which combines threads into streams."""

    _frame_class = VDIFFrame

    def __init__(self, fh_raw, header0, thread_ids):
        try:
            sample_rate = header0.bandwidth * 2
        except:
            sample_rate = None
        super(VDIFStreamBase, self).__init__(
            fh_raw=fh_raw, header0=header0, nchan=header0.nchan,
            bps=header0.bps, thread_ids=thread_ids, sample_rate=sample_rate)

    def __repr__(self):
        return ("<{s.__class__.__name__} name={s.name} offset={s.offset}\n"
                "    nthread={s.nthread}, "
                "samples_per_frame={s.samples_per_frame}, nchan={s.nchan},\n"
                "    station={h.station}, (start) time={h.time},\n"
                "    bandwidth={h.bandwidth}, complex_data={c}, "
                "bps={h.bps}, edv={h.edv}>"
                .format(s=self, h=self.header0,
                        c=self.header0['complex_data']))


class VDIFStreamReader(VDIFStreamBase):
    """VLBI VDIF format reader.

    This wrapper is allows one to access a VDIF file as a continues series of
    samples.  Invalid data are marked, but possible gaps in the data stream
    are not yet filled in.

    Parameters
    ----------
    name : str
        file name
    thread_ids: list of int
        Specific threads to read.  By default, all threads are read.
    """
    def __init__(self, raw, thread_ids=None, nthread=None):
        if isinstance(raw, io.BufferedReader):
            if not isinstance(raw, VDIFFileReader):
                raw = VDIFFileReader(raw)
        else:
            raw = VDIFFileReader(io.open(raw, mode='rb'))
        # We use the very first header in the file, since in some VLBA files
        # not all the headers have the right time.  Hopefully, the first is
        # least likely to have problems...
        header = VDIFHeader.fromfile(raw)
        # Now also read the first frameset, since we need to know how many
        # threads there are, and what the frameset size is.
        raw.seek(0)
        self._frameset = raw.read_frameset(thread_ids)
        if thread_ids is None:
            thread_ids = [fr['thread_id'] for fr in self._frameset.frames]
        self._framesetsize = raw.tell()
        super(VDIFStreamReader, self).__init__(raw, header, thread_ids)

    @lazyproperty
    def header1(self):
        raw_offset = self.fh_raw.tell()
        self.fh_raw.seek(-self.header0.framesize, 2)
        header1 = find_frame(self.fh_raw, template_header=self.header0,
                             maximum=10*self.header0.framesize, forward=False)
        self.fh_raw.seek(raw_offset)
        if header1 is None:
            raise TypeError("Corrupt VDIF? No frame in last {0} bytes."
                            .format(10*self.header0.framesize))
        return header1

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

            out = np.empty((self.nthread, count, self.nchan),
                           dtype=self._frameset.dtype).transpose(1, 0, 2)
        else:
            count = out.shape[0]
            squeeze = False

        offset0 = self.offset
        while count > 0:
            dt, frame_nr, sample_offset = self.tell(unit='frame_info')
            if(dt != self._frameset['seconds'] - self.header0['seconds'] or
               frame_nr != self._frameset['frame_nr']):
                # Read relevant frame (possibly reusing data array from
                # previous frame set).
                self._read_frame_set(fill_value, out=self._frameset._data)
                assert dt == (self._frameset['seconds'] -
                              self.header0['seconds'])
                assert frame_nr == self._frameset['frame_nr']

            data = self._frameset.data.transpose(1, 0, 2)
            # Copy relevant data from frame into output.
            nsample = min(count, self.samples_per_frame - sample_offset)
            sample = self.offset - offset0
            out[sample:sample + nsample] = data[sample_offset:
                                                sample_offset + nsample]
            self.offset += nsample
            count -= nsample

        # Ensure pointer is at right place.

        return out.squeeze() if squeeze else out

    def _read_frame_set(self, fill_value=0., out=None):
        self.fh_raw.seek(self.offset // self.samples_per_frame *
                         self._framesetsize)
        self._frameset = self.fh_raw.read_frameset(self.thread_ids,
                                                   edv=self.header0.edv)
        # Convert payloads to data array.
        data = self._frameset.todata(data=out)
        for frame, datum in zip(self._frameset.frames, data):
            if frame['invalid_data']:
                datum[...] = fill_value
        return data


class VDIFStreamWriter(VDIFStreamBase):
    """VLBI VDIF format writer.

    Parameters
    ----------
    raw : filehandle, or name.
        Should be a VDIFFileWriter or BufferedWriter instance.
        If a name, will get opened for writing binary data.
    nthread : int
        number of threads the VLBI data has (e.g., 2 for 2 polarisations)
    header : VDIFHeader
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
    station : 2 characters
        Or unsigned 2-byte integer.
    edv : 1, 3, or 4

    For edv = 1, 3, or 4, in addition, a required keyword is

    bandwidth : Quantity in Hz
        Or 'sampling_unit' + 'sample_rate'.

    For other edv, one requires

    framerate : number of frames per second.
    """
    def __init__(self, raw, nthread=1, header=None, **kwargs):
        if not isinstance(raw, io.BufferedWriter):
            raw = io.open(raw, mode='wb')
        if not isinstance(raw, VDIFFileWriter):
            raw = VDIFFileWriter(raw)
        if header is None:
            header = VDIFHeader.fromvalues(**kwargs)
        super(VDIFStreamWriter, self).__init__(raw, header, range(nthread))
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
                self.fh_raw.write_frameset(self._data, self._header)

            self.offset += nsample
            count -= nsample

    def close(self):
        extra = self.offset % self.samples_per_frame
        if extra != 0:
            warnings.warn("Closing with partial buffer remaining."
                          "Writing padded frame, marked as invalid.")
            self.write(np.zeros((extra, self.nthread, self.nchan)),
                       invalid_data=True)
            assert self.offset % self.samples_per_frame == 0
        return super(VDIFStreamWriter, self).close()


def open(name, mode='rs', *args, **kwargs):
    """Open VLBI VDIF format file for reading or writing.

    Opened as a binary file, one gets a standard file handle, but with
    methods to read/write a frame or frameset.  Opened as a stream, the file
    handler is wrapped, allowing access to it as a series of samples.

    Parameters
    ----------
    name : str
        File name
    mode : str ('rb', 'wb', 'rs', or 'ws')
        Whether to open for reading or writing, and as a regular binary file
        or as a stream (stream is default).

    Additional arguments when opening the file as a stream:

    For reading
    -----------
    thread_ids: list of int
        Specific threads to read.  By default, all threads are read.

    For writing
    -----------
    nthread : int
        number of threads the VLBI data has (e.g., 2 for 2 polarisations)
    header : VDIFHeader
        Header for the first frame, holding time information, etc.
        (or keywords that can be used to construct a header).

    Returns
    -------
    Filehandler : VDIFFileReader or VDIFFileWriter instance (binary) or
       VDIFStreamReader or VDIFStreamWriter instance (stream)

    Raises
    ------
    ValueError if an unsupported mode is chosen.
    """
    if 'w' in mode:
        fh = VDIFFileWriter(io.open(name, 'wb'))
        return fh if 'b' in mode else VDIFStreamWriter(fh, *args, **kwargs)
    elif 'r' in mode:
        fh = VDIFFileReader(io.open(name, 'rb'))
        return fh if 'b' in mode else VDIFStreamReader(fh, *args, **kwargs)
    else:
        raise ValueError("Only support opening VDIF file for reading "
                         "or writing (mode='r' or 'w').")


def find_frame(fh, template_header=None, framesize=None, maximum=None,
               forward=True):
    """Look for the first occurrence of a frame, from the current position.

    Search for a valid header at a given position which is consistent with
    `other_header` or with a header a framesize ahead.   Note that the latter
    turns out to be an unexpectedly weak check on real data!
    """
    if template_header:
        framesize = template_header.framesize

    if maximum is None:
        maximum = 2 * framesize
    # Loop over chunks to try to find the frame marker.
    file_pos = fh.tell()
    # First check whether we are right at a frame marker (usually true).
    if template_header:
        try:
            header = VDIFHeader.fromfile(fh, verify=True)
            if template_header.same_stream(header):
                fh.seek(-header.size, 1)
                return header
        except:
            pass

    if forward:
        iterate = range(file_pos, file_pos + maximum)
    else:
        iterate = range(file_pos, file_pos - maximum, -1)
    for frame in iterate:
        fh.seek(frame)
        try:
            header1 = VDIFHeader.fromfile(fh, verify=True)
        except(AssertionError, IOError, EOFError):
            continue

        if template_header:
            if template_header.same_stream(header1):
                fh.seek(frame)
                return header1
            continue

        # if no comparison header given, get header from a frame further up or
        # down and check those are consistent.
        fh.seek(frame + (framesize if forward else -framesize))
        try:
            header2 = VDIFHeader.fromfile(fh, verify=True)
        except AssertionError:
            continue
        except:
            break

        if(header2.same_stream(header1) and
           abs(header2.seconds - header1.seconds) <= 1 and
           abs(header2['frame_nr'] - header1['frame_nr']) <= 1):
            fh.seek(frame)
            return header1

    # Didn't find any frame.
    fh.seek(file_pos)
    return None


# NOT USED ANY MORE
def get_thread_ids(infile, framesize, searchsize=None):
    """
    Get the number of threads and their ID's in a vdif file.
    """
    if searchsize is None:
        searchsize = 1024 * framesize

    n_total = searchsize // framesize

    thread_ids = set()
    for n in range(n_total):
        infile.seek(n * framesize)
        try:
            thread_ids.add(VDIFHeader.fromfile(infile)['thread_id'])
        except:
            break

    return thread_ids
