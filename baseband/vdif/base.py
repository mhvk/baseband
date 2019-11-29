# Licensed under the GPLv3 - see LICENSE
from collections import namedtuple
import warnings
import bisect

import numpy as np
import astropy.units as u
from astropy.utils import lazyproperty

from ..vlbi_base.base import (make_opener, VLBIFileBase, VLBIFileReaderBase,
                              VLBIStreamBase, VLBIStreamReaderBase,
                              VLBIStreamWriterBase)
from .header import VDIFHeader
from .frame import VDIFFrame, VDIFFrameSet
from .file_info import VDIFFileReaderInfo


__all__ = ['VDIFFileReader', 'VDIFFileWriter', 'VDIFStreamBase',
           'VDIFStreamReader', 'VDIFStreamWriter', 'open']

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


class RawOffsets:
    def __init__(self, frame_nr=[], offset=[]):
        self.frame_nr = frame_nr
        self.offset = offset

    def get(self, frame_nr):
        if not self.frame_nr:
            return 0
        index = bisect.bisect_right(self.frame_nr, frame_nr)
        return 0 if index == 0 else self.offset[index - 1]

    def add(self, frame_nr, offset):
        index = bisect.bisect_right(self.frame_nr, frame_nr)
        if index < len(self) and self.offset[index] == offset:
            self.frame_nr[index] = frame_nr
        else:
            self.frame_nr.insert(index, frame_nr)
            self.offset.insert(index, offset)

    def __len__(self):
        return len(self.frame_nr)

    def __repr__(self):
        return ('{0}(frame_nr={1}, offset={2})'
                .format(type(self).__name__, self.frame_nr, self.offset))


class VDIFFileReader(VLBIFileReaderBase):
    """Simple reader for VDIF files.

    Wraps a binary filehandle, providing methods to help interpret the data,
    such as `read_frame`, `read_frameset` and `get_frame_rate`.

    Parameters
    ----------
    fh_raw : filehandle
        Filehandle of the raw binary data file.

    """
    info = VDIFFileReaderInfo()

    def read_header(self, edv=None, verify=True):
        """Read a single header from the file.

        Parameters
        ----------
        edv : int, optional
            The expected extended data version for the VDIF Header.  If `None`,
            use that of the frame.  (Passing it in slightly improves file
            integrity checking.)
        verify : bool, optional
            Whether to do basic checks of header integrity.  Default: `True`.

        Returns
        -------
        header : `~baseband.vdif.VDIFHeader`
        """
        return VDIFHeader.fromfile(self.fh_raw, edv=edv, verify=verify)

    def read_frame(self, edv=None, verify=True):
        """Read a single frame (header plus payload).

        Parameters
        ----------
        edv : int, optional
            The expected extended data version for the VDIF Header.  If `None`,
            use that of the frame.  (Passing it in slightly improves file
            integrity checking.)
        verify : bool, optional
            Whether to do basic checks of frame integrity.  Default: `True`.

        Returns
        -------
        frame : `~baseband.vdif.VDIFFrame`
            With ``.header`` and ``.data`` properties that return the
            :class:`~baseband.vdif.VDIFHeader` and data encoded in the frame,
            respectively.
        """
        return VDIFFrame.fromfile(self.fh_raw, edv=edv, verify=verify)

    def read_frameset(self, thread_ids=None, edv=None, verify=True):
        """Read a single frame (header plus payload).

        Parameters
        ----------
        thread_ids : list, optional
            The thread ids that should be read.  If `None` (default), read all
            threads.
        edv : int, optional
            The expected extended data version for the VDIF Header.  If `None`,
            use that of the first frame.  (Passing it in slightly improves file
            integrity checking.)
        verify : bool, optional
            Whether to do basic checks of frame integrity.  Default: `True`.

        Returns
        -------
        frameset : :class:`~baseband.vdif.VDIFFrameSet`
            With ``.headers`` and ``.data`` properties that return a list of
            :class:`~baseband.vdif.VDIFHeader` and the data encoded in the
            frame set, respectively.
        """
        return VDIFFrameSet.fromfile(self.fh_raw, thread_ids, edv=edv,
                                     verify=verify)

    def get_frame_rate(self):
        """Determine the number of frames per second.

        This method first tries to determine the frame rate by looking for
        the highest frame number in the first second of data.  If that fails,
        it attempts to extract the sample rate from the header.

        Returns
        -------
        frame_rate : `~astropy.units.Quantity`
            Frames per second.
        """
        try:
            return super().get_frame_rate()
        except Exception as exc:
            with self.temporary_offset():
                try:
                    self.seek(0)
                    header = self.read_header()
                    return (header.sample_rate
                            / header.samples_per_frame).to(u.Hz).round()
                except Exception:
                    pass
            raise exc

    def find_header(self, template_header=None, frame_nbytes=None, edv=None,
                    maximum=None, forward=True):
        """Find the nearest header from the current position.

        Search for a valid header at a given position which is consistent with
        ``template_header`` or with a header a frame size ahead.   Note that
        the latter turns out to be an unexpectedly weak check on real data!

        If successful, the file pointer is left at the start of the header.

        Parameters
        ----------
        template_header : `~baseband.vdif.VDIFHeader`
            If given, used to infer the frame size and EDV.
        frame_nbytes : int
            Frame size in bytes, used if ``template_header`` is not given.
        edv : int
            EDV of the header, used if ``template_header`` is not given.
        maximum : int, optional
            Maximum number of bytes forward to search through.
            Default: twice the frame size.
        forward : bool, optional
            Seek forward if `True` (default), backward if `False`.

        Returns
        -------
        header : :class:`~baseband.vdif.VDIFHeader` or None
            Retrieved VDIF header, or `None` if nothing found.
        """
        fh = self.fh_raw
        # Obtain current pointer position.
        file_pos = fh.tell()
        if template_header is not None:
            edv = template_header.edv
            # First check whether we are right at a frame marker (often true).
            try:
                header = VDIFHeader.fromfile(fh, edv=edv, verify=True)
            except(AssertionError, OSError, EOFError):
                pass
            else:
                if template_header.same_stream(header):
                    fh.seek(file_pos)
                    return header
            # If we're not at frame marker, obtain frame size and get
            # searching.
            frame_nbytes = template_header.frame_nbytes

        if maximum is None:
            maximum = 2 * frame_nbytes

        # Generate file pointer positions to test.
        nbytes = fh.seek(0, 2)
        if forward:
            iterate = range(file_pos, min(file_pos + maximum - 31,
                                          nbytes - frame_nbytes + 1))
        else:
            iterate = range(min(file_pos, nbytes - frame_nbytes),
                            max(file_pos - maximum, -1), -1)
        # Loop over chunks to try to find the frame marker.
        for frame in iterate:
            fh.seek(frame)
            try:
                header = VDIFHeader.fromfile(fh, edv=edv, verify=True)
            except AssertionError:
                continue

            if (header.frame_nbytes != frame_nbytes
                    or (template_header
                        and not template_header.same_stream(header))):
                # CPython optimizations will mark this as uncovered, even
                # though it is. See
                # https://bitbucket.org/ned/coveragepy/issues/198/continue-marked-as-not-covered
                continue  # pragma: no cover

            # Also check header from a frame up or down.
            if ((forward or frame < frame_nbytes)
                    and frame < nbytes - frame_nbytes - 32):
                next_frame = frame + frame_nbytes
            elif frame > frame_nbytes:
                next_frame = frame - frame_nbytes
            else:
                # No choice, there are no other frames.
                fh.seek(frame)
                return header

            fh.seek(next_frame)
            try:
                comparison = VDIFHeader.fromfile(fh, edv=header.edv,
                                                 verify=True)
            except AssertionError:
                continue

            if header.same_stream(comparison):
                fh.seek(frame)
                return header

        # Didn't find any frame.
        fh.seek(file_pos)
        return None


class VDIFFileWriter(VLBIFileBase):
    """Simple writer for VDIF files.

    Adds `write_frame` and `write_frameset` methods to the basic VLBI
    binary file wrapper.
    """

    def write_frame(self, data, header=None, **kwargs):
        """Write a single frame (header plus payload).

        Parameters
        ----------
        data : `~numpy.ndarray` or `~baseband.vdif.VDIFFrame`
            If an array, a ``header`` should be given, which will be used to
            get the information needed to encode the array, and to construct
            the VDIF frame.
        header : `~baseband.vdif.VDIFHeader`
            Can instead give keyword arguments to construct a header.  Ignored
            if ``data`` is a `~baseband.vdif.VDIFFrame` instance.
        **kwargs
            If ``header`` is not given, these are used to initialize one.
        """
        if not isinstance(data, VDIFFrame):
            data = VDIFFrame.fromdata(data, header, **kwargs)
        return data.tofile(self.fh_raw)

    def write_frameset(self, data, header=None, **kwargs):
        """Write a single frame set (headers plus payloads).

        Parameters
        ----------
        data : `~numpy.ndarray` or :class:`~baseband.vdif.VDIFFrameSet`
            If an array, a header should be given, which will be used to
            get the information needed to encode the array, and to construct
            the VDIF frame set.
        header : :class:`~baseband.vdif.VDIFHeader`, list of same
            Can instead give keyword arguments to construct a header.  Ignored
            if ``data`` is a :class:`~baseband.vdif.VDIFFrameSet` instance.
            If a list, should have a length matching the number of threads in
            ``data``; if a single header, ``thread_ids`` corresponding
            to the number of threads are generated automatically.
        **kwargs
            If ``header`` is not given, these are used to initialize one.
        """
        if not isinstance(data, VDIFFrameSet):
            data = VDIFFrameSet.fromdata(data, header, **kwargs)
        return data.tofile(self.fh_raw)


class VDIFStreamBase(VLBIStreamBase):
    """Base for VDIF streams."""

    _sample_shape_maker = namedtuple('SampleShape', 'nthread, nchan')

    def __init__(self, fh_raw, header0, sample_rate=None, nthread=1,
                 squeeze=True, subset=(), fill_value=0., verify=True):
        samples_per_frame = header0.samples_per_frame

        super().__init__(
            fh_raw=fh_raw, header0=header0, sample_rate=sample_rate,
            samples_per_frame=samples_per_frame,
            unsliced_shape=(nthread, header0.nchan), bps=header0.bps,
            complex_data=header0['complex_data'], squeeze=squeeze,
            subset=subset, fill_value=fill_value, verify=verify)

        self._frame_rate = int((self.sample_rate / self.samples_per_frame)
                               .to(u.Hz).round().value)

    def _get_time(self, header):
        """Get time from a header.

        This passes on sample rate, since not all VDIF headers can calculate
        it.
        """
        return header.get_time(frame_rate=self.sample_rate
                               / self.samples_per_frame)

    def __repr__(self):
        return ("<{s.__class__.__name__} name={s.name} offset={s.offset}\n"
                "    sample_rate={s.sample_rate},"
                " samples_per_frame={s.samples_per_frame},\n"
                "    sample_shape={s.sample_shape},\n"
                "    bps={h.bps}, complex_data={s.complex_data},"
                " edv={h.edv}, station={h.station},\n"
                "    {sub}start_time={s.start_time}>"
                .format(s=self, h=self.header0,
                        sub=('subset={0}, '.format(self.subset)
                             if self.subset else '')))


class VDIFStreamReader(VDIFStreamBase, VLBIStreamReaderBase):
    """VLBI VDIF format reader.

    Allows access to a VDIF file as a continuous series of samples.

    Parameters
    ----------
    fh_raw : filehandle
        Filehandle of the raw VDIF stream.
    sample_rate : `~astropy.units.Quantity`, optional
        Number of complete samples per second, i.e. the rate at which each
        channel in each thread is sampled.  If `None` (default), will be
        inferred from the header or by scanning one second of the file.
    squeeze : bool, optional
        If `True` (default), remove any dimensions of length unity from
        decoded data.
    subset : indexing object or tuple of objects, optional
        Specific components of the complete sample to decode (after possible
        squeezing).  If a single indexing object is passed, it selects threads.
        If a tuple is passed, the first selects threads and the second selects
        channels.  If the tuple is empty (default), all components are read.
    fill_value : float or complex, optional
        Value to use for invalid or missing data. Default: 0.
    verify : bool, optional
        Whether to do basic checks of frame integrity when reading.  The first
        frameset of the stream is always checked.  Default: `True`.
    """

    def __init__(self, fh_raw, sample_rate=None, squeeze=True, subset=(),
                 fill_value=0., verify=True):
        fh_raw = VDIFFileReader(fh_raw)
        # We read the first frameset, since we need to know how many threads
        # there are, and what the frameset size is.
        # Note that frameset keeps the first header, since in some VLBA files
        # not all the headers have the right time.  Hopefully, the first is
        # least likely to have problems...
        frameset0 = fh_raw.read_frameset()
        # Sometimes the first frameset is incomplete, so read another one,
        # but ignore any errors just in case we have a very short file.
        offset = fh_raw.tell()
        try:
            frameset = fh_raw.read_frameset()
        except Exception:
            frameset = frameset0

        if len(frameset) <= len(frameset0):
            frameset = frameset0
            self._frameset_nbytes = offset
            self._raw_offsets = RawOffsets()
            fh_raw.seek(0)
        else:
            self._frameset_nbytes = fh_raw.tell() - offset
            self._raw_offsets = RawOffsets(frame_nr=[0], offset=[offset])
            fh_raw.seek(offset)

        header0 = frameset.header0
        thread_ids = [fr['thread_id'] for fr in frameset.frames]
        super().__init__(
            fh_raw, header0, sample_rate=sample_rate,
            nthread=len(frameset.frames), squeeze=squeeze, subset=subset,
            fill_value=fill_value, verify=verify)
        # Check whether we are reading only some threads.  This is somewhat
        # messy since normally we apply the whole subset to the whole data,
        # but here we need to split it up in the part that selects specific
        # threads, which we use to selectively read, and the rest, which we
        # do post-decoding.
        if self.subset and (len(thread_ids) > 1 or not self.squeeze):
            # Select the thread ids we want using first part of subset.
            thread_ids = np.array(thread_ids)[self.subset[0]]
            # Use squeese in case subset[0] uses broadcasting, and
            # atleast_1d to ensure single threads get upgraded to a list,
            # which is needed by the VDIFFrameSet reader.
            self._thread_ids = np.atleast_1d(thread_ids.squeeze()).tolist()
            # Reload the frame set, now only including the requested threads.
            frameset = fh_raw.read_frameset(self._thread_ids)
            # Since we have subset the threads already, we now need to
            # determine a new subset that takes this into account.
            if thread_ids.shape == ():
                # If we indexed with a scalar, we're meant to remove that
                # dimension. If we squeeze, this happens automatically, but if
                # not, we need to do it explicitly (FrameSet does not do it).
                new_subset0 = () if self.squeeze else (0,)
            elif len(self._thread_ids) == 1 and self.squeeze:
                # If we want a single remaining thread, undo the squeeze.
                new_subset0 = (np.newaxis,)
            else:
                # Just pass on multiple threads or unsqueezed single ones.
                new_subset0 = (slice(None),)

            self._frameset_subset = new_subset0 + self.subset[1:]
        else:
            # We either have no subset or we have a single thread that
            # will be squeezed away, so the subset is fine as is.
            self._frameset_subset = self.subset
            self._thread_ids = thread_ids

        self._frameset = frameset

    @lazyproperty
    def _last_header(self):
        """Last header of the file."""
        # Go to end of file.
        with self.fh_raw.temporary_offset() as fh_raw:
            fh_raw.seek(0, 2)
            raw_size = fh_raw.tell()
            # Find first header with same thread_id going backward.
            found = False
            # Set maximum as twice number of frames in frameset.
            maximum = 2 * self._frameset_nbytes
            while not found:
                fh_raw.seek(-self.header0.frame_nbytes, 1)
                last_header = fh_raw.find_header(
                    template_header=self.header0,
                    maximum=maximum, forward=False)
                if last_header is None or (raw_size - fh_raw.tell() > maximum):
                    raise ValueError("corrupt VDIF? No thread_id={0} frame "
                                     "in last {1} bytes."
                                     .format(self.header0['thread_id'],
                                             maximum))

                found = last_header['thread_id'] == self.header0['thread_id']

        return last_header

    def _squeeze_and_subset(self, data):
        # Overwrite VLBIStreamReaderBase version, since the threads part of
        # subset has already been used.
        if self.squeeze:
            data = data.reshape(data.shape[:1]
                                + tuple(sh for sh in data.shape[1:] if sh > 1))
        if self._frameset_subset:
            data = data[(slice(None),) + self._frameset_subset]

        return data

    def _read_frame(self, index):
        raw_corr = self._raw_offsets.get(index)
        raw_offset = index * self._frameset_nbytes + raw_corr
        self.fh_raw.seek(raw_offset)
        try:
            frameset = self.fh_raw.read_frameset(self._thread_ids,
                                                 edv=self.header0.edv,
                                                 verify=self.verify)
            assert ((frameset['seconds'] - self.header0['seconds'])
                    * self._frame_rate
                    + frameset['frame_nr'] - self.header0['frame_nr']) == index
        except (IOError, AssertionError) as exc:
            # Something went wrong.
            dt, frame_nr = divmod(index, self._frameset_nbytes)
            msg = ('Problem loading frame set at seconds={0}, frame_nr={1}.'
                   .format(self.header0['seconds'] + dt, frame_nr))
            # See if we're in the right place.  First ensure we have a header.
            self.fh_raw.seek(raw_offset)
            header = self.fh_raw.find_header(self.header0, forward=True)
            if header is None:
                warnings.warn(msg + ' Cannot find header nearby.')
                raise exc

            def frame_index(header):
                return ((header['seconds'] - self.header0['seconds'])
                        * self._frame_rate
                        + header['frame_nr'] - self.header0['frame_nr'])

            # Don't yet know how to deal with excess data.
            if frame_index(header) < index:
                warnings.warn(msg + ' There appears to be excess data.')
                raise exc

            # Go backward until we find previous frame
            while frame_index(header) >= index:
                raw_pos = self.fh_raw.tell()
                header1 = header
                self.fh_raw.seek(-1, 1)
                header = self.fh_raw.find_header(self.header0, forward=False)
                if header is None:
                    raise exc

            # Move back to position of last good header (header1).
            self.fh_raw.seek(raw_pos)

            if frame_index(header1) < index:
                # Whole frame set missing?
                self._raw_offsets.add(index, None)
                self._raw_offsets.add(frame_index(header1),
                                      raw_pos - frame_index(header1)
                                      * self._frameset_nbytes)
                warnings.warn(msg + ' The frame seems to be missing.')
                raise NotImplementedError("Cannot deal with missing frames.")

            assert frame_index(header1) == index, \
                'at this point, we either errored or have a good header.'

            if raw_pos != raw_offset:
                msg += ' Stream off by {0} bytes.'.format(raw_offset - raw_pos)
                self._raw_offsets.add(index,
                                      raw_pos - index * self._frameset_nbytes)

            # Try again to read it, however many threads there are.
            # TODO: this somewhat duplicates FrameSet.fromfile; possibly
            # move code there.
            # TODO: remove limitation that threads need to be together.
            frames = {}
            previous = None
            while True:
                raw_pos = self.fh_raw.tell()
                try:
                    frame = self.fh_raw.read_frame(edv=self.header0.edv)
                except EOFError:
                    # End of file while reading a frame; we're done here.
                    next_header = None
                    break
                except AssertionError:
                    # Frame is not OK.
                    assert previous is not None, \
                        'first frame should be readable if fully on disk'

                    # Go back to after previous payload and try finding
                    # next header.
                    self.fh_raw.seek(raw_pos - header1.payload_nbytes)
                    next_header = self.fh_raw.find_header(self.header0)

                    # If no header was found, give up.  The previous frame
                    # was likely bad too, so delete it.
                    if next_header is None:
                        del frames[previous]
                        break

                    # If the next header is not exactly a frame away from
                    # where we were trying to read, the previous frame was
                    # likely bad, so discard it.
                    if self.fh_raw.tell() != raw_pos + header1.frame_nbytes:
                        del frames[previous]

                    # Stop if the next header is from a different frame.
                    if next_header['frame_nr'] != frame_nr:
                        break

                else:
                    # Successfully read frame.  If it is not of the requested
                    # set, rewind and break out of the loop.
                    if frame['frame_nr'] != frame_nr:
                        next_header = frame.header
                        self.fh_raw.seek(raw_pos)
                        break

                    # Do we have a good frame, giving a new thread?
                    previous = frame['thread_id']
                    if previous in frames:
                        msg += (' Duplicate thread {0} found; discarding.'
                                .format(previous))
                        del frames[previous]
                    else:
                        # Looks like it, though may still be discarded
                        # if the next frame is not readable.
                        frames[previous] = frame

            # If the next header is of the next frame, set up the raw
            # offset (which likely will be needed, saving some time).
            if (next_header is not None
                    and frame_index(next_header) == index + 1):
                self._raw_offsets.add(index + 1,
                                      self.fh_raw.tell() - (index + 1)
                                      * self._frameset_nbytes)

            # Create invalid frame template, using header1, which is
            # guaranteed to be from this set, as a base.
            # It is copied to make it mutable without any risk of messing up
            # possibly memory mapped data.
            invalid_frame = self._frameset.frames[0].__class__(
                header1.copy(), self._frameset.frames[0].payload)
            invalid_frame.valid = False

            frame_list = []
            missing = []
            for thread in self._thread_ids:
                if thread in frames:
                    frame_list.append(frames[thread])
                else:
                    missing.append(thread)
                    invalid_frame.header['thread_id'] = thread
                    frame_list.append(invalid_frame)

            if missing:
                if frames == {}:
                    msg += ' Failed to get any threads; all set to invalid.'
                else:
                    msg += (' Thread(s) {0} missing; set to invalid.'
                            .format(missing))

            warnings.warn(msg)

            frameset = self._frameset.__class__(frame_list)

        frameset.fill_value = self.fill_value
        return frameset


class VDIFStreamWriter(VDIFStreamBase, VLBIStreamWriterBase):
    """VLBI VDIF format writer.

    Encodes and writes sequences of samples to file.

    Parameters
    ----------
    fh_raw : filehandle
        Which will write filled sets of frames to storage.
    header0 : :class:`~baseband.vdif.VDIFHeader`
        Header for the first frame, holding time information, etc.  Can instead
        give keyword arguments to construct a header (see ``**kwargs``).
    sample_rate : `~astropy.units.Quantity`
        Number of complete samples per second, i.e. the rate at which each
        channel in each thread is sampled.  For EDV 1 and 3, can
        alternatively set ``sample_rate`` within the header.
    nthread : int, optional
        Number of threads (e.g., 2 for 2 polarisations).  Default: 1.
    squeeze : bool, optional
        If `True` (default), `write` accepts squeezed arrays as input, and
        adds any dimensions of length unity.
    **kwargs
        If no header is given, an attempt is made to construct one from these.
        For a standard header, this would include the following.

    --- Header keywords : (see :meth:`~baseband.vdif.VDIFHeader.fromvalues`)

    time : `~astropy.time.Time`
        Start time of the file.  Can instead pass on ``ref_epoch`` and
        ``seconds``.
    nchan : int, optional
        Number of channels (default: 1).  Note: different numbers of channels
        per thread is not supported.
    complex_data : bool, optional
        Whether data are complex.  Default: `False`.
    bps : int, optional
        Bits per elementary sample, i.e. per real or imaginary component for
        complex data.  Default: 1.
    samples_per_frame : int
        Number of complete samples per frame.  Can alternatively use
        ``frame_length``, the number of 8-byte words for header plus payload.
        For some EDV, this number is fixed (e.g., ``frame_length=629`` for
        ``edv=3``, which corresponds to 20000 real 2-bit samples per frame).
    station : 2 characters, optional
        Station ID.  Can also be an unsigned 2-byte integer.  Default: 0.
    edv : {`False`, 0, 1, 2, 3, 4, 0xab}
        Extended Data Version.
    """

    def __init__(self, fh_raw, header0=None, sample_rate=None, nthread=1,
                 squeeze=True, **kwargs):
        fh_raw = VDIFFileWriter(fh_raw)
        if header0 is None:
            if sample_rate is not None:
                kwargs['sample_rate'] = sample_rate
            header0 = VDIFHeader.fromvalues(**kwargs)

        # If header was passed but not sample_rate, extract sample_rate.
        if sample_rate is None:
            try:
                sample_rate = header0.sample_rate
            except AttributeError:
                raise ValueError("the sample rate must be passed either "
                                 "explicitly, or through the header if it "
                                 "can be stored there.")

        super().__init__(fh_raw, header0, sample_rate=sample_rate,
                         nthread=nthread, squeeze=squeeze)
        # Set sample rate in the header, if it's possible, and not set already.
        try:
            header_sample_rate = self.header0.sample_rate
        except AttributeError:
            pass
        else:
            if header_sample_rate == 0:
                self.header0.sample_rate = self.sample_rate
            assert self.header0.sample_rate == self.sample_rate

        self._frameset = VDIFFrameSet.fromdata(
            np.zeros((self.samples_per_frame,) + self._unsliced_shape,
                     dtype=np.complex64 if self.complex_data else np.float32),
            self.header0)

    def _make_frame(self, index):
        dt, frame_nr = divmod(index + self.header0['frame_nr'],
                              self._frame_rate)
        seconds = self.header0['seconds'] + dt
        # Reuse frameset.
        self._frameset['seconds'] = seconds
        self._frameset['frame_nr'] = frame_nr
        return self._frameset


open = make_opener('VDIF', globals(), doc="""
--- For reading a stream : (see :class:`~baseband.vdif.base.VDIFStreamReader`)

sample_rate : `~astropy.units.Quantity`, optional
    Number of complete samples per second, i.e. the rate at which each channel
    in each thread is sampled.  If `None` (default), will be inferred from the
    header or by scanning one second of the file.
squeeze : bool, optional
    If `True` (default), remove any dimensions of length unity from
    decoded data.
subset : indexing object or tuple of objects, optional
    Specific components of the complete sample to decode (after possible
    squeezing).  If a single indexing object is passed, it selects threads.
    If a tuple is passed, the first selects threads and the second selects
    channels.  If the tuple is empty (default), all components are read.
fill_value : float or complex, optional
    Value to use for invalid or missing data. Default: 0.
verify : bool, optional
    Whether to do basic checks of frame integrity when reading.  The first
    frameset of the stream is always checked.  Default: `True`.

--- For writing a stream : (see :class:`~baseband.vdif.base.VDIFStreamWriter`)

header0 : `~baseband.vdif.VDIFHeader`
    Header for the first frame, holding time information, etc.  Can instead
    give keyword arguments to construct a header (see ``**kwargs``).
sample_rate : `~astropy.units.Quantity`
    Number of complete samples per second, i.e. the rate at which each
    channel in each thread is sampled.  For EDV 1 and 3, can alternatively set
    ``sample_rate`` within the header.
nthread : int, optional
    Number of threads (e.g., 2 for 2 polarisations).  Default: 1.
squeeze : bool, optional
    If `True` (default), writer accepts squeezed arrays as input, and adds any
    dimensions of length unity.
file_size : int or None, optional
    When writing to a sequence of files, the maximum size of one file in bytes.
    If `None` (default), the file size is unlimited, and only the first
    file will be written to.
**kwargs
    If the header is not given, an attempt will be made to construct one
    with any further keyword arguments.  See
    :class:`~baseband.vdif.base.VDIFStreamWriter`.

Notes
-----
One can also pass to ``name`` a list, tuple, or subclass of
`~baseband.helpers.sequentialfile.FileNameSequencer`.  For writing to multiple
files, the ``file_size`` keyword must be passed or only the first file will be
written to.  One may also pass in a `~baseband.helpers.sequentialfile` object
(opened in 'rb' mode for reading or 'w+b' for writing), though for typical use
cases it is practically identical to passing in a list or template.
""")
