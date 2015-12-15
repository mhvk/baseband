# Licensed under the GPLv3 - see LICENSE.rst
import io

import numpy as np
from astropy.utils import lazyproperty
import astropy.units as u

from ..vlbi_base.base import (VLBIStreamBase, VLBIStreamReaderBase,
                              VLBIStreamWriterBase)
from .header import VDIFHeader
from .frame import VDIFFrame, VDIFFrameSet


__all__ = ['VDIFFileReader', 'VDIFFileWriter', 'VDIFStreamBase',
           'VDIFStreamReader', 'VDIFStreamWriter', 'open',
           'find_frame']

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
    file reader :class:`~io.BufferedReader`.
    """
    def read_frame(self):
        """Read a single frame (header plus payload).

        Returns
        -------
        frame : `~baseband.vdif.VDIFFrame`
            With ``.header`` and ``.data`` properties that return the
            :class:`~baseband.vdif.VDIFHeader` and data encoded in the frame,
            respectively.
        """
        return VDIFFrame.fromfile(self)

    def read_frameset(self, thread_ids=None, sort=True, edv=None, verify=True):
        """Read a single frame (header plus payload).

        Parameters
        ----------
        thread_ids : list, optional
            The thread ids that should be read.  By default, read all threads.
        sort : bool, optional
            Whether to sort the frames by thread_id.  Default: True.
        edv : int, optional
            The expected extended data version for the VDIF Header.  If not
            given, use that of the first frame.  (Passing it in slightly
            improves file integrity checking.)
        verify : bool, optional
            Whether to do (light) sanity checks on the header. Default: True.

        Returns
        -------
        frameset : :class:`~baseband.vdif.VDIFFrameSet`
            With ``.headers`` and ``.data`` properties that return a list of
            :class:`~baseband.vdif.VDIFHeaders` and the data encoded in the
            frame set, respectively.
        """
        return VDIFFrameSet.fromfile(self, thread_ids, sort=sort, edv=edv,
                                     verify=verify)


class VDIFFileWriter(io.BufferedWriter):
    """Simple writer for VDIF files.

    Adds ``write_frame`` and ``write_frameset`` methods to the basic binary
    file writer :class:`~io.BufferedWriter`.
    """
    def write_frame(self, data, header=None, **kwargs):
        """Write a single frame (header plus payload).

        Parameters
        ----------
        data : array or `~baseband.vdif.VDIFFrame`
            If an array, a `header` should be given, which will be used to
            get the information needed to encode the array, and to construct
            the VDIF frame.
        header : `~baseband.vdif.VDIFHeader`, optional
            Ignored if `data` is a VDIF Frame.
        **kwargs
            If no `header` is given, an attempt is made to initialize one
            using keywords arguments.
        """
        if not isinstance(data, VDIFFrame):
            data = VDIFFrame.fromdata(data, header, **kwargs)
        return data.tofile(self)

    def write_frameset(self, data, header=None, **kwargs):
        """Write a frame set (headers plus payloads).

        Parameters
        ----------
        data : array or :class:`~baseband.vdif.VDIFFrameSet`
            If an array, a header should be given, which will be used to
            get the information needed to encode the array, and to construct
            the VDIF frame set.
        header : :class:`~baseband.vdif.VDIFHeader`, list of same, optional
            Ignored if `data` is a :class:`~baseband.vdif.VDIFFrameSet`
            instance.  If a list, should have a length matching the number of
            threads in `data`; if a single header, ``thread_ids`` corresponding
            to the number of threads are generated automatically.
        **kwargs
            If no `header` is given, remaining keywords are used to attempt
            to initiate a single header.
        """
        if not isinstance(data, VDIFFrameSet):
            data = VDIFFrameSet.fromdata(data, header, **kwargs)
        return data.tofile(self)


class VDIFStreamBase(VLBIStreamBase):
    """VDIF file wrapper, which combines threads into streams."""

    _frame_class = VDIFFrame

    def __init__(self, fh_raw, header0, thread_ids, frames_per_second=None,
                 sample_rate=None):
        if frames_per_second is None and sample_rate is None:
            try:
                frames_per_second = int(header0.framerate.to(u.Hz).value)
            except:
                pass

        super(VDIFStreamBase, self).__init__(
            fh_raw=fh_raw, header0=header0, nchan=header0.nchan,
            bps=header0.bps, thread_ids=thread_ids,
            samples_per_frame=header0.samples_per_frame,
            frames_per_second=frames_per_second,
            sample_rate=sample_rate)

    def __repr__(self):
        return ("<{s.__class__.__name__} name={s.name} offset={s.offset}\n"
                "    nthread={s.nthread}, "
                "samples_per_frame={s.samples_per_frame}, nchan={s.nchan},\n"
                "    frames_per_second={s.frames_per_second}, "
                "complex_data={c}, bps={h.bps}, edv={h.edv},\n"
                "    station={h.station}, (start) time={h.time}>"
                .format(s=self, h=self.header0,
                        c=self.header0['complex_data']))


class VDIFStreamReader(VDIFStreamBase, VLBIStreamReaderBase):
    """VLBI VDIF format reader.

    This wrapper is allows one to access a VDIF file as a continues series of
    samples.  Invalid data are marked, but possible gaps in the data stream
    are not yet filled in.

    Parameters
    ----------
    raw : `~baseband.vdif.VDIFFileReader` instance
        file handle of the raw VDIF stream
    thread_ids: list of int, optional
        Specific threads to read.  By default, all threads are read.
    frames_per_second : int
        Needed to calculate timestamps. If not given, will be inferred from
        ``sample_rate``, EDV bandwidth, or by scanning the file.
    sample_rate : `~astropy.units.Quantity`, optional
        Rate at which each channel in each thread is sampled.
    """
    def __init__(self, raw, thread_ids=None, frames_per_second=None,
                 sample_rate=None):
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
        super(VDIFStreamReader, self).__init__(raw, header, thread_ids,
                                               frames_per_second, sample_rate)

    @lazyproperty
    def header1(self):
        """Last header of the file."""
        raw_offset = self.fh_raw.tell()
        # Go to end of file.
        self.fh_raw.seek(0, 2)
        raw_size = self.fh_raw.tell()
        # Find first header with same thread_id going backward.
        found = False
        while not found:
            self.fh_raw.seek(-self.header0.framesize, 1)
            header1 = find_frame(self.fh_raw, template_header=self.header0,
                                 maximum=10*self.header0.framesize,
                                 forward=False)
            if header1 is None:
                raise TypeError("Corrupt VDIF? No thread_id={0} frame in last "
                                "{1} bytes."
                                .format(self.header0['thread_id'],
                                        raw_size - self.fh_raw.tell()))

            found = header1['thread_id'] == self.header0['thread_id']

        self.fh_raw.seek(raw_offset)
        return header1

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
            dt, frame_nr, sample_offset = self._frame_info()
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
        return self._frameset.todata(data=out, invalid_data_value=fill_value)


class VDIFStreamWriter(VDIFStreamBase, VLBIStreamWriterBase):
    """VLBI VDIF format writer.

    Parameters
    ----------
    raw : `~baseband.vdif.VDIFFileWriter`
        Which will write filled sets of frames to storage.
    nthread : int
        Number of threads the VLBI data has (e.g., 2 for 2 polarisations).
        Default is 1.
    frames_per_second : int, optional
        Needed to calculate timestamps. Can also give ``sample_rate``.
        Only needed if the EDV does not have bandwidth information.
    sample_rate : `~astropy.units.Quantity`, optional
        Rate at which each channel in each thread is sampled.
    header : :class:`~baseband.vdif.VDIFHeader`, optional
        Header for the first frame, holding time information, etc.
    **kwargs
        If no header is give, an attempt is made to construct the header from
        these.  For a standard header, this would include the following.

    --- Header keywords : (see :meth:`~baseband.vdif.VDIFHeader.fromvalues`)

    time : `~astropy.time.Time`
        As an alternative, one can pass on ``ref_epoch`` and ``seconds``.
    nchan : int, optional
        Number of FFT channels within stream (default 1).
        Note: that different # of channels per thread is not supported.
    complex_data : bool
        Whether data is complex
    bps : int
        Bits per sample (or real, imaginary component).
    samples_per_frame : int
        Number of complete samples in a given frame.  As an alternative, use
        ``frame_length``, the number  of long words for header plus payload.
        For some edv, this number is fixed (e.g., ``frame_length=629`` for
        edv=3, which corresponds to 20000 real 2-bit samples per frame).
    station : 2 characters
        Or unsigned 2-byte integer.
    edv : {`False`, 0, 1, 2, 3, 4, 0xab}
        Extended Data Version.
    bandwidth : `~astropy.units.Quantity`
        In frequency units.  Sufficient for `edv` 1, 3, or 4 to determine the
        frames per second.
    """
    def __init__(self, raw, nthread=1, frames_per_second=None,
                 sample_rate=None, header=None, **kwargs):
        if header is None:
            header = VDIFHeader.fromvalues(**kwargs)
        super(VDIFStreamWriter, self).__init__(
            raw, header, range(nthread), frames_per_second=frames_per_second,
            sample_rate=sample_rate)
        try:
            header_framerate = self.header0.framerate
        except:
            pass
        else:
            if header_framerate == 0:
                header.framerate = self.frames_per_second * u.Hz
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
            dt, frame_nr, sample_offset = self._frame_info()
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


def open(name, mode='rs', *args, **kwargs):
    """Open VLBI VDIF format file for reading or writing.

    Opened as a binary file, one gets a standard file handle, but with
    methods to read/write a frame or frameset.  Opened as a stream, the file
    handler is wrapped, allowing access to it as a series of samples.

    Parameters
    ----------
    name : str
        File name
    mode : {'rb', 'wb', 'rs', or 'ws'}, optional
        Whether to open for reading or writing, and as a regular binary file
        or as a stream (default is reading a stream).
    **kwargs :
        Additional arguments when opening the file as a stream

    --- For reading : (see :class:`VDIFStreamReader`)

    thread_ids : list of int, optional
        Specific threads to read.  By default, all threads are read.
    frames_per_second : int
        Needed to calculate timestamps. If not given, will be inferred from
        ``sample_rate``, EDV bandwidth, or by scanning the file.
    sample_rate : `~astropy.units.Quantity`, optional
        Rate at which each channel in each thread is sampled.

    --- For writing : (see :class:`VDIFStreamWriter`)

    nthread : int
        Number of threads the VLBI data has (e.g., 2 for 2 polarisations).
    frames_per_second : int, optional
        Needed to calculate timestamps. Can also give ``sample_rate``.
        Only needed if the EDV does not have bandwidth information.
    sample_rate : `~astropy.units.Quantity`, optional
        Rate at which each channel in each thread is sampled.
    header : `~baseband.vdif.VDIFHeader`, optional
        Header for the first frame, holding time information, etc.
    **kwargs
        If the header is not given, an attempt will be made to construct one
        with any further keyword arguments.  See :class:`VDIFStreamWriter`.

    Returns
    -------
    Filehandle
        A :class:`VDIFFileReader` or :class:`VDIFFileWriter` instance (binary),
        or a :class:`VDIFStreamReader` or :class:`VDIFStreamWriter` instance
        (stream).
    """
    if 'w' in mode:
        if not hasattr(name, 'write'):
            name = io.open(name, 'wb')
        fh = VDIFFileWriter(name)
        return fh if 'b' in mode else VDIFStreamWriter(fh, *args, **kwargs)
    elif 'r' in mode:
        if not hasattr(name, 'read'):
            name = io.open(name, 'rb')
        fh = VDIFFileReader(name)
        return fh if 'b' in mode else VDIFStreamReader(fh, *args, **kwargs)
    else:
        raise ValueError("Only support opening VDIF file for reading "
                         "or writing (mode='r' or 'w').")


def find_frame(fh, template_header=None, framesize=None, maximum=None,
               forward=True):
    """Look for the first occurrence of a frame, from the current position.

    Search for a valid header at a given position which is consistent with
    ``other_header`` or with a header a framesize ahead.   Note that the latter
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
