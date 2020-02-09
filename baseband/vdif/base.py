# Licensed under the GPLv3 - see LICENSE
import warnings
from collections import namedtuple

import numpy as np
import astropy.units as u
from astropy.utils import lazyproperty

from ..vlbi_base.base import (make_opener, VLBIFileBase, VLBIFileReaderBase,
                              VLBIStreamBase, VLBIStreamReaderBase,
                              VLBIStreamWriterBase, HeaderNotFoundError)
from .header import VDIFHeader
from .payload import VDIFPayload
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
        edv : int, False, or None, optional
            Extended data version.  If `False`, a legacy header is used.
            If `None` (default), it is determined from the header.  (Given it
            explicitly is mostly useful for a slight speed-up.)
        verify : bool, optional
            Whether to do basic verification of integrity.  Default: `True`.

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
            use that of the first frame.  (Passing it in slightly improves file
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

    def get_thread_ids(self, check=2):
        """Determine the number of threads in the VDIF file.

        The file is presumed to be positioned at the start of a header.
        Usually, it suffices to just seek to the start of the file, but
        if not, use `~baseband.vdif.base.VDIFFileReader.find_header`.

        Parameters
        ----------
        check : int, optional
            Number of extra frames to check.  Frame sets are scanned until
            the number of thread IDs found no longer increases for ``check``
            frames.

        Returns
        -------
        thread_ids : list
            Sorted list of all thread ids encountered in the frames scanned.
        """
        with self.temporary_offset():
            header = header0 = self.read_header()
            try:
                thread_ids = set()
                n_check = 1
                while n_check > 0:
                    frame_nr = header['frame_nr']
                    n_thread = len(thread_ids)
                    while header['frame_nr'] == frame_nr:
                        thread_ids.add(header['thread_id'])
                        self.seek(header.payload_nbytes, 1)
                        header = self.read_header(edv=header0.edv)
                        assert header0.same_stream(header)

                    if len(thread_ids) > n_thread:
                        n_check = check
                    else:
                        n_check -= 1
            except EOFError:
                # Hack: let through very short files (like our samples).
                if self.seek(0, 2) > ((check+2) * len(thread_ids)
                                      * header0.frame_nbytes):
                    raise

        return sorted(thread_ids)

    def find_header(self, pattern=None, *, edv=None, mask=None,
                    frame_nbytes=None, offset=0,
                    forward=True, maximum=None, check=1):
        """Find the nearest header from the current position.

        Search for a valid header at a given position which is consistent with
        ``pattern`` and/or with a header a frame size ahead.  Note
        that the search is much slower if no pattern is given, as at every
        position it is tried to read a header, and then check for another one
        one frame ahead.  It helps to pass in ``edv`` and ``frame_nbytes``
        (if known).

        If successful, the file pointer is left at the start of the header.

        Parameters
        ----------
        pattern : `~baseband.vdif.VDIFHeader`, array of byte, or compatible
            If given, used for a direct search.
        edv : int
            EDV of the header, used if ``pattern`` is not given.
        mask : array of byte, bytes, iterable of int, string or int
            Bit mask for the pattern, with 1 indicating a given bit will
            be used the comparison.  Only used with ``pattern`` and not
            needed if ``pattern`` is a header.
        frame_nbytes : int, optional
            Frame size in bytes.  Defaults to the frame size in any header
            passed in.
        offset : int, optional
            Offset from the frame start that the pattern occurs.  Any
            offsets inferred from masked entries are added to this (hence,
            no offset needed when a header is passed in as ``pattern``,
            nor is an offset needed for a full search).
        forward : bool, optional
            Seek forward if `True` (default), backward if `False`.
        maximum : int, optional
            Maximum number of bytes to search away from the present location.
            Default: search twice the frame size if given, otherwise 10000
            (extra bytes to avoid partial patterns will be added).
            Use 0 to check only at the current position.
        check : int or tuple of int, optional
            Frame offsets where another header should be present.
            Default: 1, i.e., a sync pattern should be present one
            frame after the one found (independent of ``forward``),
            thus helping to guarantee the frame is not corrupted.

        Returns
        -------
        header : :class:`~baseband.vdif.VDIFHeader`
            Retrieved VDIF header.

        Raises
        ------
        ~baseband.vlbi_base.base.HeaderNotFoundError
            If no header could be located.
        AssertionError
            If the header did not pass verification.
        """
        if pattern is not None:
            locations = self.locate_frames(
                pattern, mask=mask, frame_nbytes=frame_nbytes, offset=offset,
                forward=forward, maximum=maximum, check=check)
            if not locations:
                raise HeaderNotFoundError('could not locate a a nearby frame.')
            self.seek(locations[0])
            with self.temporary_offset():
                return self.read_header(edv=getattr(pattern, 'edv', None))

        # Try reading headers at a set of locations.
        if maximum is None:
            maximum = 10000 if frame_nbytes is None else 2 * frame_nbytes

        file_pos = self.tell()
        # Generate file pointer positions to test.
        if forward:
            iterate = range(file_pos, file_pos+maximum+1)
        else:
            iterate = range(file_pos, max(file_pos-maximum-1, -1), -1)
        # Loop over all of them to try to find the frame marker.
        for frame in iterate:
            self.seek(frame)
            try:
                header = self.read_header(edv=edv)
            except Exception:
                continue

            if (frame_nbytes is not None
                    and frame_nbytes != header.frame_nbytes):
                continue

            # Possible hit!  Try if there are other headers right around.
            self.seek(frame)
            try:
                return self.find_header(header, maximum=0, check=check)
            except Exception:
                continue

        self.seek(file_pos)
        raise HeaderNotFoundError("could not locate a nearby header.")


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

    def _get_time(self, header):
        """Get time from a header.

        This passes on sample rate, since not all VDIF headers can calculate
        it.
        """
        return header.get_time(frame_rate=self._frame_rate)

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
                 fill_value=0., verify='fix'):
        fh_raw = VDIFFileReader(fh_raw)
        # We read the very first header, hoping this is the right one
        # (in some VLBA files not all the headers have the right time).
        header0 = fh_raw.read_header()
        # Next, we determine how many threads there are, and use those
        # to calculate the frameset size.  We on purpose do *not* just read
        # a frame set, since sometimes the first one is short (see gh-359).
        fh_raw.seek(0)
        thread_ids = fh_raw.get_thread_ids()
        nthread = len(thread_ids)
        super().__init__(
            fh_raw, header0, sample_rate=sample_rate,
            nthread=nthread, squeeze=squeeze, subset=subset,
            fill_value=fill_value, verify=verify)
        self._raw_offsets.frame_nbytes *= nthread

        # Check whether we are reading only some threads.  This is somewhat
        # messy since normally we apply the whole subset to the whole data,
        # but here we need to split it up in the part that selects specific
        # threads, which we use to selectively read, and the rest, which we
        # do post-decoding.
        if self.subset and (nthread > 1 or not self.squeeze):
            # Select the thread ids we want using first part of subset.
            thread_ids = np.array(thread_ids)[self.subset[0]]
            # Use squeese in case subset[0] uses broadcasting, and
            # atleast_1d to ensure single threads get upgraded to a list,
            # which is needed by the VDIFFrameSet reader.
            self._thread_ids = np.atleast_1d(thread_ids.squeeze()).tolist()
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

    @lazyproperty
    def _last_header(self):
        """Last header of the file."""
        # Go to end of file.
        with self.fh_raw.temporary_offset() as fh_raw:
            maximum = 2 * self._raw_offsets.frame_nbytes
            # Find first header with same thread_id going backward.
            fh_raw.seek(-self.header0.frame_nbytes, 2)
            locations = fh_raw.locate_frames(self.header0, forward=False,
                                             maximum=maximum, check=(-1, 1))
            for location in locations:
                fh_raw.seek(location)
                try:
                    header = fh_raw.read_header(edv=self.header0.edv)
                except Exception:
                    continue

                if header['thread_id'] == self.header0['thread_id']:
                    return header

            raise HeaderNotFoundError(
                "corrupt VDIF? No thread_id={0} frame in last {1} bytes."
                .format(self.header0['thread_id'], maximum))

    def _squeeze_and_subset(self, data):
        # Overwrite VLBIStreamReaderBase version, since the threads part of
        # subset has already been used.
        if self.squeeze:
            data = data.reshape(data.shape[:1]
                                + tuple(sh for sh in data.shape[1:] if sh > 1))
        if self._frameset_subset:
            data = data[(slice(None),) + self._frameset_subset]

        return data

    # Overrides to deal with framesets instead of frames.
    def _fh_raw_read_frame(self):
        return self.fh_raw.read_frameset(self._thread_ids,
                                         edv=self.header0.edv,
                                         verify=self.verify)

    def _tell_frame(self, frame):
        return int(round((frame['seconds'] - self.header0['seconds'])
                         * self._frame_rate.to_value(u.Hz)
                         + frame['frame_nr'] - self.header0['frame_nr']))

    def _bad_frame(self, index, frameset, exc):
        # Duplication of base class, but able to deal with missing
        # frames inside a frame set.
        if (frameset is not None and self._tell_frame(frameset) == index
                and index == self._tell_frame(self._last_header)):
            # If we got an exception because we're trying to read beyond the
            # last frame, the frame is almost certainly OK, so keep it.
            return frameset

        if self.verify != 'fix':
            raise exc

        # If the frameset does contain the right number of frames, but all
        # are invalid, assume not just the data but also the frame number
        # and seconds might be wrong, i.e., just proceed with it.
        # TODO: make this an option for a specific type of fixing!
        if (frameset is not None
                and len(frameset.frames) == len(self._thread_ids)
                and not any(frame.valid for frame in frameset.frames)):
            return frameset

        msg = 'problem loading frame set {}.'.format(index)

        # Where should we be?
        raw_offset = self._seek_frame(index)

        # See if we're in the right place.  First ensure we have a header.
        # Here, it is more important that it is a good one than that we go
        # too far, so we insist on two consistent frames after it, as well
        # as a good one before to guard against corruption of the start of
        # the VDIF header.
        self.fh_raw.seek(raw_offset)
        try:
            header = self.fh_raw.find_header(
                self.header0, forward=True, check=(-1, 1, 2),
                maximum=3*self.header0.frame_nbytes)
        except HeaderNotFoundError:
            exc.args += (msg + ' Cannot find header nearby.',)
            raise exc

        # Don't yet know how to deal with excess data.
        header_index = self._tell_frame(header)
        if header_index < index:
            exc.args += (msg + ' There appears to be excess data.',)
            raise exc

        # Go backward until we find previous frame, storing offsets
        # as we go.  We again increase the maximum since we may need
        # to jump over a bad bit.  We slightly relax our search pattern.
        while header_index >= index:
            raw_pos = self.fh_raw.tell()
            header1 = header
            header1_index = header_index
            if raw_pos <= 0:
                break

            self.fh_raw.seek(-1, 1)
            try:
                header = self.fh_raw.find_header(
                    self.header0, forward=False,
                    maximum=4*self.header0.frame_nbytes,
                    check=(-1, 1))
            except HeaderNotFoundError:
                exc.args += (msg + ' Could not find previous index.',)
                raise exc

            header_index = self._tell_frame(header)
            if header_index < header1_index:
                # While we are at it: if we pass an index boundary,
                # update the list of known indices.
                self._raw_offsets[header1_index] = raw_pos

        # Move back to position of last good header (header1).
        self.fh_raw.seek(raw_pos)

        # Create the header we will use below for constructing the
        # frameset.  Usually, this is guaranteed to be from this set,
        # but we also use it to create a new header for a completely
        # messed up frameset below.  It is copied to make it mutable
        # without any risk of messing up possibly memory mapped data.
        header = header1.copy()

        if header1_index > index:
            # Ouch, whole frame set missing!
            msg += ' The frame set seems to be missing altogether.'
            # Set up to construct a complete missing frame
            # (after the very long else clause).
            # TODO: just use the writer's _make_frame??
            frames = {}
            dt, frame_nr = divmod(index + self.header0['frame_nr'],
                                  int(self._frame_rate.to_value(u.Hz)))
            header['frame_nr'] = frame_nr
            header['seconds'] = self.header0['seconds'] + dt
        else:
            assert header1_index == index, \
                'at this point, we should have a good header.'
            # This header is the first one of its set.
            if raw_pos != raw_offset:
                msg += ' Stream off by {0} bytes.'.format(raw_offset
                                                          - raw_pos)
                # Above, we should have added information about
                # this index in our offset table.
                assert raw_pos == self._raw_offsets[index]

            # Try again to read it, however many threads there are.
            # TODO: this somewhat duplicates FrameSet.fromfile; possibly
            #       move code there.
            # TODO: Or keep track of header locations above.
            # TODO: remove limitation that threads need to be together.
            frames = {}
            previous = False
            frame_nr = header1['frame_nr']
            while True:
                raw_pos = self.fh_raw.tell()
                try:
                    frame = self.fh_raw.read_frame(edv=self.header0.edv)
                    assert header.same_stream(frame.header)
                    # Check seconds as well, as a sanity check this is
                    # a real header. (We do allow at this point it to be
                    # the next frame, hence the second can increase by 1.
                    assert 0 <= (frame['seconds'] - header['seconds']) <= 1
                except EOFError:
                    # End of file while reading a frame; we're done here.
                    next_header = None
                    break
                except AssertionError:
                    # Frame is not OK.
                    assert previous is not False, \
                        ('first frame should be readable if fully on disk,'
                         ' since we found one correct header.')

                    # Go back to after previous payload and try finding
                    # next header.  It can be before where we tried above,
                    # if some bytes in the previous payload were missing.
                    self.fh_raw.seek(raw_pos - header.payload_nbytes)
                    try:
                        next_header = self.fh_raw.find_header(self.header0)
                        # But sometimes a header is re-found even when
                        # there isn't really one (e.g., because one of the
                        # first bytes, defining seconds, is missing).
                        # Don't ever retry the same one!
                        if self.fh_raw.tell() == raw_pos:
                            self.fh_raw.seek(1, 1)
                            next_header = self.fh_raw.find_header(self.header0)
                    except HeaderNotFoundError:
                        # If no header was found, give up.  The previous frame
                        # was likely bad too, so delete it.
                        if previous is not None:
                            del frames[previous]
                        next_header = None
                        break

                    # If the next header is not exactly a frame away from
                    # where we were trying to read, the previous frame was
                    # likely bad, so discard it.
                    if self.fh_raw.tell() != raw_pos + header.frame_nbytes:
                        if previous is not None:
                            del frames[previous]
                        previous = None

                    # Stop if the next header is from a different frame.
                    if next_header['frame_nr'] != frame_nr:
                        break

                else:
                    # Successfully read frame.  If not of the requested
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
                    and self._tell_frame(next_header) == index + 1):
                self._raw_offsets[index+1] = self.fh_raw.tell()

        # Create invalid frame template,
        invalid_payload = VDIFPayload(
            np.zeros(header.payload_nbytes // 4, '<u4'), header)
        invalid_frame = VDIFFrame(header, invalid_payload, valid=False)

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
                msg += ' All threads set to invalid.'
            else:
                msg += (' Thread(s) {0} missing; set to invalid.'
                        .format(missing))

        warnings.warn(msg)
        frameset = VDIFFrameSet(frame_list)
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
                              int(self._frame_rate.to_value(u.Hz)))
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
