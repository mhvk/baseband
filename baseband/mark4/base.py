# Licensed under the GPLv3 - see LICENSE.rst
import numpy as np

from ..vlbi_base.base import (make_opener, VLBIFileBase, VLBIStreamReaderBase,
                              VLBIStreamWriterBase)
from .header import Mark4Header
from .payload import Mark4Payload
from .frame import Mark4Frame


__all__ = ['Mark4FileReader', 'Mark4FileWriter', 'Mark4StreamReader',
           'Mark4StreamWriter', 'open']

# Look-up table for the number of bits in a byte.
nbits = ((np.arange(256)[:, np.newaxis] >> np.arange(8) & 1)
         .sum(1).astype(np.int16))


class Mark4FileReader(VLBIFileBase):
    """Simple reader for Mark 4 files.

    Adds ``read_frame`` and ``find_frame`` methods to the VLBI file wrapper.
    """

    def read_frame(self, ntrack, decade):
        """Read a single frame (header plus payload).

        Parameters
        ----------
        ntrack : int
            Number of Mark 4 bitstreams.
        decade : int
            Decade the observations were taken (needed to remove ambiguity in
            the Mark 4 time stamp).

        Returns
        -------
        frame : `~baseband.mark4.Mark4Frame`
            With ``.header`` and ``.data`` properties that return the
            :class:`~baseband.mark4.Mark4Header` and data encoded in the frame,
            respectively.
        """
        return Mark4Frame.fromfile(self.fh_raw, ntrack=ntrack, decade=decade)

    def find_frame(self, ntrack, maximum=None, forward=True):
        """Look for the first occurrence of a frame, from the current position.

        The search is for the following pattern:

        * 32*tracks bits set at offset bytes
        * 1*tracks bits unset before offset
        * 32*tracks bits set at offset+2500*tracks bytes

        This reflects 'sync_pattern' of 0xffffffff for a given header and one
        a frame ahead, which is in word 2, plus the lsb of word 1, which is
        'system_id'.

        Parameters
        ----------
        ntrack : int
            Number of "tape tracks".
        maximum : int, optional
            Maximum number of bytes forward to search through.
            Default is the framesize (20000 * ntrack // 8).
        forward : bool, optional
            Whether to search forwards or backwards.  Default is forwards.

        Returns
        -------
        offset : int
            Byte offset of the first frame.
        """
        fh = self.fh_raw
        nset = np.ones(32 * ntrack // 8, dtype=np.int16)
        nunset = np.ones(ntrack // 8, dtype=np.int16)
        framesize = ntrack * 2500
        file_pos = fh.tell()
        fh.seek(0, 2)
        filesize = fh.tell()
        if filesize < framesize:
            fh.seek(file_pos)
            return None

        if maximum is None:
            maximum = 2 * framesize
        # Loop over chunks to try to find the frame marker.
        step = framesize // 2
        # read a bit more at every step to ensure we don't miss a "split"
        # header
        block = step + 160 * ntrack // 8
        if forward:
            iterate = range(max(min(file_pos, filesize - block), 0),
                            max(min(file_pos + maximum, filesize - block + 1),
                                1),
                            step)
        else:
            iterate = range(min(max(file_pos - step, 0), filesize - block),
                            min(max(file_pos - step - maximum - 1, -1),
                                filesize - block),
                            -step)
        for frame in iterate:
            fh.seek(frame)

            data = np.frombuffer(fh.read(block), dtype=np.uint8)
            assert len(data) == block
            # Find header pattern.
            databits1 = nbits[data]
            nosync = np.convolve(databits1[len(nunset):] < 6, nset, 'valid')
            nolow = np.convolve(databits1[:-len(nset)] > 1, nunset, 'valid')
            wrong = nosync + nolow
            possibilities = np.where(wrong == 0)[0]
            # check candidates by seeing whether there is a sync word
            # a framesize ahead. (Note: loop can be empty)
            for possibility in possibilities[::1 if forward else -1]:
                # real start of possible header.
                frame_start = frame + possibility - 63 * ntrack // 8
                if (forward and frame_start < file_pos or
                        not forward and frame_start > file_pos):
                    continue
                # check there is a header following this.
                check = frame_start + framesize
                if check >= filesize - 32 * 2 * ntrack // 8 - len(nunset):
                    # but do before this one if we're beyond end of file.
                    check = frame_start - framesize
                    if check < 0:  # assume OK if only one frame fits in file.
                        if frame_start + framesize > filesize:
                            continue
                        else:
                            break

                fh.seek(check + 32 * 2 * ntrack // 8)
                check_data = np.frombuffer(fh.read(len(nunset)),
                                           dtype=np.uint8)
                databits2 = nbits[check_data]
                if np.all(databits2 >= 6):
                    break  # got it!

            else:  # None of them worked, so do next block
                continue

            fh.seek(frame_start)
            return frame_start

        fh.seek(file_pos)
        return None

    def find_frame_and_ntrack(self, maximum=None, forward=True):
        """Finds start of the next frame, and number of tracks ``ntrack``.

        Uses ``find_frame`` to look for the first occurrence of a frame from
        the current position while cycling through all supported ``ntrack``
        values.

        Parameters
        ----------
        maximum : int, optional
            Maximum number of bytes forward to search through.
            Default is the framesize (20000 * ntrack // 8).
        forward : bool, optional
            Whether to search forwards or backwards.  Default is forwards.

        Returns
        -------
        offset : int
            Byte offset of the first frame.
        ntrack : int
            Number of tracks.
        """
        # Currently only 32 and 64-track frames supported.
        for nt in 32, 64:
            frame_start = self.find_frame(nt, maximum=maximum, forward=forward)
            if frame_start:
                break

        if frame_start is None:
            nt = None

        return frame_start, nt

    def find_header(self, template_header=None, ntrack=None, decade=None,
                    maximum=None, forward=True):
        """Look for the first occurrence of a frame, from the current position.

        Read the header at that location and return it.
        The file pointer is left at the start of the header.

        Parameters
        ----------
        template_header : :class:`~baseband.mark4.Mark4Header`, optional
            Template Mark 4 header, from which `ntrack` and `decade` are
            extracted.
        ntrack : int, optional
            Number of tracks used to store the data.  Required if
            ``template_header`` is ``None``.
        decade : int, optional
            Decade the observations were taken (needed to remove ambiguity in
            the Mark 4 time stamp).  Required if ``template_header`` is
            ``None``.
        maximum : int, optional
            Maximum number of bytes to search through.  Default is twice the
            framesize.
        forward : bool, optional
            Seek forward if ``True`` (default), backward if ``False``.

        Returns
        -------
        header : :class:`~baseband.mark4.Mark4Header`, or None
            Retrieved Mark 4 header, or ``None`` if nothing found.
        """
        if template_header is not None:
            ntrack = template_header.ntrack
            decade = template_header.decade
        offset = self.find_frame(ntrack, maximum, forward)
        if offset is None:
            return None
        header = Mark4Header.fromfile(self.fh_raw, ntrack=ntrack,
                                      decade=decade)
        self.fh_raw.seek(offset)
        return header


class Mark4FileWriter(VLBIFileBase):
    """Simple writer for Mark 4 files.

    Adds ``write_frame`` method to the VLBI binary file wrapper.
    """
    def write_frame(self, data, header=None, **kwargs):
        """Write a single frame (header plus payload).

        Parameters
        ----------
        data : array or `~baseband.mark4.Mark4Frame`
            If an array, a header should be given, which will be used to
            get the information needed to encode the array, and to construct
            the Mark 4 frame.
        header : `~baseband.mark4.Mark4Header`
            Ignored if payload is a :class:`~baseband.mark4.Mark4Frame`
            instance.
        **kwargs :
            If no header is given, these are used to initialize one.
        """
        if not isinstance(data, Mark4Frame):
            data = Mark4Frame.fromdata(data, header, **kwargs)
        return data.tofile(self.fh_raw)


class Mark4StreamReader(VLBIStreamReaderBase, Mark4FileReader):
    """VLBI Mark 4 format reader.

    This wrapper is allows one to access a Mark 4 file as a continues series
    of samples.  Note that possible gaps in the data stream are not filled in,
    though parts of the data stream replaced by header values are filled in.

    Parameters
    ----------
    fh_raw : `~baseband.mark4.Mark4FileReader`
        file handle to the raw Mark 4 data stream.
    ntrack : int, or None
        Number of tracks used to store the data.  If ``None``, will attempt to
        determine it by scanning the file.
    decade : int, or `~astropy.time.Time`
        Decade the observations were taken (needed to remove ambiguity in the
        Mark 4 time stamp).
    thread_ids: list of int, optional
        Specific threads/channels to read.  By default, all are read.
    frames_per_second : int, optional
        Needed to calculate timestamps. If not given, will be inferred from
        ``sample_rate``, or by scanning the file.
    sample_rate : `~astropy.units.Quantity`, optional
        Rate at which each thread is sampled (bandwidth * 2; frequency units).
    squeeze : bool, optional
        If `True` (default), remove any dimensions of length unity from
        decoded data.
    """

    _frame_class = Mark4Frame

    def __init__(self, fh_raw, ntrack=None, decade=None, thread_ids=None,
                 frames_per_second=None, sample_rate=None, squeeze=True):
        # Pre-set fh_raw, so FileReader methods work
        # TODO: move this to StreamReaderBase?
        self.fh_raw = fh_raw
        # Find offset for first header, and ntrack if not specified.
        if ntrack is None:
            self.offset0, ntrack = self.find_frame_and_ntrack()
        else:
            self.offset0 = self.find_frame(ntrack=ntrack)
        if self.offset0 is None:
            raise Exception('Could not find the first occurrence of a header.'
                            'If automatic ntrack detection was used, try'
                            'passing in an explicit ntrack.')
        # If decade is an astropy.time.Time object, extract decade.
        try:
            decade = decade.__index__()
        except AttributeError:
            decade = 10 * (decade.datetime.year // 10)
        self._frame = self.read_frame(ntrack, decade)
        self._frame_data = None
        self._frame_nr = None
        header = self._frame.header
        sample_shape = (Mark4Payload._sample_shape_maker(len(thread_ids)) if
                        thread_ids else self._frame.payload.sample_shape)
        super(Mark4StreamReader, self).__init__(
            fh_raw, header0=header, sample_shape=sample_shape,
            bps=header.bps, complex_data=False, thread_ids=thread_ids,
            samples_per_frame=header.samples_per_frame,
            frames_per_second=frames_per_second, sample_rate=sample_rate,
            squeeze=squeeze)

    @staticmethod
    def _get_frame_rate(fh, header_template):
        """Returns the number of frames in one second of data.

        Parameters
        ----------
        fh : io.BufferedReader
            Binary file handle.
        header_template : header class or instance
            Definition or instance of file format's header class.

        Returns
        -------
        fps : int
            Frames per second.

        Notes
        -----

        Unlike VLBIStreamReaderBase._get_frame_rate, this function reads
        only two consecutive frames, extracting their timestamps to determine
        how much time has elapsed.  It will return an EOFError if there is
        only one frame.
        """
        oldpos = fh.tell()
        header0 = header_template.fromfile(fh, header_template.ntrack,
                                           decade=header_template.decade)
        fh.seek(header0.payloadsize, 1)
        header1 = header_template.fromfile(fh, header_template.ntrack,
                                           decade=header_template.decade)
        fh.seek(oldpos)
        # Mark 4 specification states frames-lengths range from 1.25 ms
        # to 160 ms.
        tdelta = header1.ms[0] - header0.ms[0]
        return int(np.round(1000. / tdelta))

    def read(self, count=None, fill_value=0., out=None):
        """Read count samples.

        The range retrieved can span multiple frames.

        Parameters
        ----------
        count : int
            Number of samples to read.  If omitted or negative, the whole
            file is read.  Ignored if ``out`` is given.
        fill_value : float
            Value to use for invalid or missing data.
        out : `None` or array
            Array to store the data in. If given, ``count`` will be inferred
            from the first dimension.  The other dimension should equal
            ``sample_shape``.

        Returns
        -------
        out : array of float
            The first dimension is sample-time, and the second, given by
            ``sample_shape``, is (channel,).  Any dimension of length unity is
            removed if ``self.squeeze=True``.
        """
        if out is None:
            if count is None or count < 0:
                count = self.size - self.offset

            result = np.empty(self._sample_shape + (count,),
                              dtype=self._frame.dtype).T
            out = result.squeeze() if self.squeeze else result
        else:
            count = out.shape[0]
            result = self._unsqueeze(out) if self.squeeze else out

        offset0 = self.offset
        while count > 0:
            frame_nr, sample_offset = divmod(self.offset,
                                             self.samples_per_frame)
            if frame_nr != self._frame_nr:
                self._read_frame()

            data = self._frame.data
            if self.thread_ids:
                data = data[:, self.thread_ids]
            # Copy relevant data from frame into output.
            nsample = min(count, self.samples_per_frame - sample_offset)
            sample = self.offset - offset0
            result[sample:sample + nsample] = data[sample_offset:
                                                   sample_offset + nsample]
            self.offset += nsample
            count -= nsample

        return out

    def _read_frame(self):
        frame_nr = self.offset // self.samples_per_frame
        self.fh_raw.seek(self.offset0 + frame_nr * self.header0.framesize)
        self._frame = self.read_frame(ntrack=self.header0.ntrack,
                                      decade=self.header0.decade)
        # Convert payloads to data array.
        self._frame_data = self._frame.data
        self._frame_nr = frame_nr


class Mark4StreamWriter(VLBIStreamWriterBase, Mark4FileWriter):
    """VLBI Mark 4 format writer.

    Parameters
    ----------
    raw : `~baseband.mark4.Mark4FileWriter`
        Which will write filled sets of frames to storage.
    frames_per_second : int, optional
        Needed to calculate timestamps. If not given, inferred from
        ``sample_rate``.
    sample_rate : `~astropy.units.Quantity`, optional
        Rate at which each thread is sampled (bandwidth * 2; frequency units).
    header : `~baseband.mark4.Mark4Header`
        Header for the first frame, holding start time information, etc.
    squeeze : bool, optional
        If `True` (default), ``write`` accepts squeezed arrays as input,
        and adds channel and thread dimensions if they have length unity.
    **kwargs
        If no header is give, an attempt is made to construct the header from
        these.  For a standard header, this would include the following.

    --- Header keywords : (see :meth:`~baseband.mark4.Mark4Header.fromvalues`)

    time : `~astropy.time.Time`
        Sets bcd-encoded unit year, day, hour, minute, second.
    ntrack : int
        Number of Mark 4 bitstreams (equal to number of channels times
        ``fanout`` times ``bps``)
    bps : int
        Bits per sample.
    fanout : int
        Number of tracks over which a given channel is spread out.
    """

    _frame_class = Mark4Frame

    def __init__(self, raw, frames_per_second=None, sample_rate=None,
                 header=None, squeeze=True, **kwargs):
        if header is None:
            header = Mark4Header.fromvalues(**kwargs)
        sample_shape = Mark4Payload._sample_shape_maker(header.nchan)
        super(Mark4StreamWriter, self).__init__(
            fh_raw=raw, header0=header, sample_shape=sample_shape,
            thread_ids=range(header.nchan), bps=header.bps, complex_data=False,
            samples_per_frame=(header.framesize * 8 // header.bps //
                               header.nchan),
            frames_per_second=frames_per_second, sample_rate=sample_rate,
            squeeze=squeeze)

        self._data = np.zeros((self.samples_per_frame,
                               self._sample_shape.nchan), np.float32)

    def write(self, data, invalid_data=False):
        """Write data, buffering by frames as needed.

        Parameters
        ----------
        data : array
            Piece of data to be written, with sample dimensions as given by
            ``sample_shape``. This should be properly scaled to make best use
            of the dynamic range delivered by the encoding.
        invalid_data : bool, optional
            Whether the current data is valid.  Defaults to `False`.
        """
        if self.squeeze:
            data = self._unsqueeze(data)

        assert data.shape[1] == self._sample_shape.nchan

        count = data.shape[0]
        sample = 0
        offset0 = self.offset
        frame = self._data
        while count > 0:
            frame_nr, sample_offset = divmod(self.tell(),
                                             self.samples_per_frame)
            if sample_offset == 0:
                # set up header for new frame.
                self._header = self.header0.copy()
                self._header.update(time=self.tell(unit='time'))

            if invalid_data:
                # Mark whole frame as invalid data.
                self._header['communication_error'] = True

            nsample = min(count, self.samples_per_frame - sample_offset)
            sample_end = sample_offset + nsample
            sample = self.offset - offset0
            frame[sample_offset:sample_end] = data[sample:sample + nsample]
            if sample_end == self.samples_per_frame:
                self.write_frame(self._data, self._header)

            self.offset += nsample
            count -= nsample


open = make_opener('Mark4', globals(), doc="""
--- For reading a stream : (see `~baseband.mark4.base.Mark4StreamReader`)

ntrack : int
    Number of tracks used to store the data.
thread_ids: list of int, optional
    Specific threads/channels to read.  By default, all are read.
frames_per_second : int, optional
    Needed to calculate timestamps. If not given, will be inferred from
    ``sample_rate``, or by scanning the file.
sample_rate : `~astropy.units.Quantity`, optional
    Rate at which each thread is sampled (bandwidth * 2; frequency units).
squeeze : bool, optional
    If `True` (default), remove any dimensions of length unity from
    decoded data.

--- For writing a stream : (see `~baseband.mark4.base.Mark4StreamWriter`)

frames_per_second : int, optional
    Needed to calculate timestamps. If not given, inferred from
    ``sample_rate``.
sample_rate : `~astropy.units.Quantity`, optional
    Rate at which each thread is sampled (bandwidth * 2; frequency units).
header : `~baseband.mark4.Mark4Header`
    Header for the first frame, holding time information, etc.
squeeze : bool, optional
    If `True` (default), ``write`` accepts squeezed arrays as input,
    and adds channel and thread dimensions if they have length unity.
**kwargs
    If the header is not given, an attempt will be made to construct one
    with any further keyword arguments.  See
    :class:`~baseband.mark4.base.Mark4StreamWriter`.

Returns
-------
Filehandle
    :class:`~baseband.mark4.base.Mark4FileReader` or
    :class:`~baseband.mark4.base.Mark4FileWriter` instance (binary), or
    :class:`~baseband.mark4.base.Mark4StreamReader` or
    :class:`~baseband.mark4.base.Mark4StreamWriter` instance (stream)
""")
