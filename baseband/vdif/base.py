# Licensed under the GPLv3 - see LICENSE.rst
import numpy as np
from astropy.utils import lazyproperty
import astropy.units as u
from collections import namedtuple

from ..vlbi_base.base import (make_opener, VLBIFileBase, VLBIStreamBase,
                              VLBIStreamReaderBase, VLBIStreamWriterBase)
from .header import VDIFHeader
from .frame import VDIFFrame, VDIFFrameSet


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


class VDIFFileReader(VLBIFileBase):
    """Simple reader for VDIF files.

    Adds ``read_frame`` and ``read_frameset`` methods on top of a binary
    file reader (which is wrapped as ``self.fh_raw``).
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
        return VDIFFrame.fromfile(self.fh_raw)

    def read_frameset(self, thread_ids=None, edv=None, verify=True):
        """Read a single frame (header plus payload).

        Parameters
        ----------
        thread_ids : list, optional
            The thread ids that should be read.  By default, read all threads.
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
        return VDIFFrameSet.fromfile(self.fh_raw, thread_ids, edv=edv,
                                     verify=verify)

    def find_header(self, template_header=None, framesize=None, edv=None,
                    maximum=None, forward=True):
        """Look for the first occurrence of a header, from the current position.

        Search for a valid header at a given position which is consistent with
        ``template_header`` or with a header a framesize ahead.   Note that the
        latter turns out to be an unexpectedly weak check on real data!
        """
        fh = self.fh_raw
        # Obtain current pointer position.
        file_pos = fh.tell()
        if template_header is not None:
            edv = template_header.edv
            # First check whether we are right at a frame marker (often true).
            try:
                header = VDIFHeader.fromfile(fh, edv=edv, verify=True)
            except(AssertionError, IOError, EOFError):
                pass
            else:
                if template_header.same_stream(header):
                    fh.seek(file_pos)
                    return header
            # If we're not at frame marker, obtain framesize and get searching.
            framesize = template_header.framesize

        if maximum is None:
            maximum = 2 * framesize

        # Determine file size.
        file_pos = fh.tell()
        fh.seek(0, 2)
        size = fh.tell()
        # Generate file pointer positions to test.
        if forward:
            iterate = range(file_pos, min(file_pos + maximum - 31,
                                          size - framesize + 1))
        else:
            iterate = range(min(file_pos, size - framesize),
                            max(file_pos - maximum, -1), -1)
        # Loop over chunks to try to find the frame marker.
        for frame in iterate:
            fh.seek(frame)
            try:
                header = VDIFHeader.fromfile(fh, edv=edv, verify=True)
            except AssertionError:
                continue

            if(header.framesize != framesize or
               template_header and not template_header.same_stream(header)):
                # CPython optimizations will make this as uncovered, even
                # though it is. See
                # https://bitbucket.org/ned/coveragepy/issues/198/continue-marked-as-not-covered
                continue  # pragma: no cover

            # Always also check header from a frame up.
            next_frame = frame + framesize
            if next_frame > size - 32:
                # if we're too far ahead for there to be another header,
                # check consistency with a frame below.
                next_frame = frame - framesize
                # But don't bother if we already checked with a template,
                # or if there is only one frame in the first place.
                if template_header is not None or next_frame < 0:
                    fh.seek(frame)
                    return header

            fh.seek(next_frame)
            try:
                comparison = VDIFHeader.fromfile(fh, edv=header.edv,
                                                 verify=True)
            except AssertionError:
                continue

            if comparison.same_stream(header):
                fh.seek(frame)
                return header

        # Didn't find any frame.
        fh.seek(file_pos)
        return None


class VDIFFileWriter(VLBIFileBase):
    """Simple writer for VDIF files.

    Adds ``write_frame`` and ``write_frameset`` methods to the basic VLBI
    binary file wrapper.
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
        return data.tofile(self.fh_raw)

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
        return data.tofile(self.fh_raw)


class VDIFStreamBase(VLBIStreamBase):
    """VDIF file wrapper, allowing access as a stream of data."""

    _sample_shape_maker = namedtuple('SampleShape', 'nthread, nchan')

    def __init__(self, fh_raw, header0, subset, nthread, sample_rate=None,
                 fill_value=0., squeeze=True):
        samples_per_frame = header0.samples_per_frame
        if sample_rate is None:
            try:
                sample_rate = header0.sample_rate
            except AttributeError:
                pass  # super below will scan file to get sample rate.

        super(VDIFStreamBase, self).__init__(
            fh_raw=fh_raw, header0=header0,
            unsliced_shape=(nthread, header0.nchan),
            bps=header0.bps, complex_data=header0['complex_data'],
            subset=subset, samples_per_frame=samples_per_frame,
            sample_rate=sample_rate, fill_value=fill_value, squeeze=squeeze)

    def _get_time(self, header):
        """Get time from a header.

        This passes on sample rate, since not all VDIF headers can calculate
        it.
        """
        return header.get_time(sample_rate=self.sample_rate)

    def __repr__(self):
        return ("<{s.__class__.__name__} name={s.name} offset={s.offset}\n"
                "    sample_rate={s.sample_rate},"
                " samples_per_frame={s.samples_per_frame},\n"
                "    sample_shape={s.sample_shape},\n"
                "    complex_data={s.complex_data},"
                " bps={h.bps}, edv={h.edv}, station={h.station},\n"
                "    {sub}start_time={s.start_time}>"
                .format(s=self, h=self.header0,
                        sub=('subset={0}, '.format(self.subset)
                             if self.subset else '')))


class VDIFStreamReader(VDIFStreamBase, VLBIStreamReaderBase, VDIFFileReader):
    """VLBI VDIF format reader.

    This wrapper allows one to access a VDIF file as a continues series of
    samples.  Invalid data are marked, but possible gaps in the data stream
    are not yet filled in.

    Parameters
    ----------
    fh_raw : `~baseband.vdif.base.VDIFFileReader` instance
        File handle of the raw VDIF stream
    subset : indexing object or tuple of objects, optional
        Specific components of the complete sample to decode.  If a single
        indexing object is passed, it selects threads.  If a tuple of
        objects is passed, the first selects threads and the second selects
        channels.  By default, all components are read.
    sample_rate : `~astropy.units.Quantity`, optional
        Number of complete samples per second (ie. the rate at which each
        channel in each thread is sampled).  If not given, will be inferred
        from the header or by scanning one second of the file.
    fill_value : float or complex
        Value to use for invalid or missing data. Default: 0.
    squeeze : bool, optional
        If `True` (default), remove any dimensions of length unity from
        decoded data.
    """

    def __init__(self, fh_raw, subset=(), sample_rate=None, fill_value=0.,
                 squeeze=True):
        # We use the very first header in the file, since in some VLBA files
        # not all the headers have the right time.  Hopefully, the first is
        # least likely to have problems...
        header = VDIFHeader.fromfile(fh_raw)
        # TODO: make this a bit less ugly?  Maybe don't read frameset twice?
        fh_raw.seek(0)
        # Now also read the first frameset, since we need to know how many
        # threads there are, and what the frameset size is. For this purpose,
        # pre-attach the file handle (it will just get reset anyway).
        self.fh_raw = fh_raw
        frameset = self.read_frameset()
        thread_ids = [fr['thread_id'] for fr in frameset.frames]
        self._framesetsize = fh_raw.tell()
        super(VDIFStreamReader, self).__init__(
            fh_raw, header, subset, len(frameset.frames),
            sample_rate=sample_rate, fill_value=fill_value, squeeze=squeeze)
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
            self.fh_raw.seek(0)
            frameset = self.read_frameset(self._thread_ids)
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
        self._framerate = int(round((self.sample_rate /
                                     self.samples_per_frame).to_value(u.Hz)))

    @lazyproperty
    def _last_header(self):
        """Last header of the file."""
        raw_offset = self.fh_raw.tell()
        # Go to end of file.
        self.fh_raw.seek(0, 2)
        raw_size = self.fh_raw.tell()
        # Find first header with same thread_id going backward.
        found = False
        # Set maximum as twice number of frames in frameset.
        maximum = 2 * self._framesetsize
        while not found:
            self.fh_raw.seek(-self.header0.framesize, 1)
            last_header = self.find_header(
                template_header=self.header0,
                maximum=maximum, forward=False)
            if last_header is None or raw_size - self.fh_raw.tell() > maximum:
                raise ValueError("corrupt VDIF? No thread_id={0} frame in "
                                 "last {1} bytes."
                                 .format(self.header0['thread_id'], maximum))

            found = last_header['thread_id'] == self.header0['thread_id']

        self.fh_raw.seek(raw_offset)
        return last_header

    def _squeeze_and_subset(self, data):
        # Overwrite VLBIStreamReaderBase version, since the threads part of
        # subset has already been used.
        if self.squeeze:
            data = data.reshape(data.shape[:1] +
                                tuple(sh for sh in data.shape[1:] if sh > 1))
        if self._frameset_subset:
            data = data[(slice(None),) + self._frameset_subset]

        return data

    def _read_frame(self, frame_nr):
        self.fh_raw.seek(frame_nr * self._framesetsize)
        frameset = self.read_frameset(self._thread_ids,
                                      edv=self.header0.edv)
        frameset.invalid_data_value = self.fill_value
        assert ((frameset['seconds'] - self.header0['seconds']) *
                self._framerate +
                frameset['frame_nr'] - self.header0['frame_nr']) == frame_nr
        # TODO: once framesets are sliceable, just return frameset itself.
        # For now, keep it around for testing, inspection.
        self._frameset = frameset
        return frameset.data


class VDIFStreamWriter(VDIFStreamBase, VLBIStreamWriterBase, VDIFFileWriter):
    """VLBI VDIF format writer.

    Parameters
    ----------
    raw : `~baseband.vdif.base.VDIFFileWriter`
        Which will write filled sets of frames to storage.
    nthread : int
        Number of threads the VLBI data has (e.g., 2 for 2 polarizations).
        Default is 1.
    sample_rate : `~astropy.units.Quantity`, optional
        Number of complete samples per second (ie. the rate at which each
        channel in each thread is sampled).  For EDV 1 and 3, can
        alternatively set `sample_rate` within the header passed to `header`.
    header : :class:`~baseband.vdif.VDIFHeader`, optional
        Header for the first frame, holding time information, etc.
    squeeze : bool, optional
        If `True` (default), ``write`` accepts squeezed arrays as input,
        and adds channel and thread dimensions if they have length unity.
    **kwargs
        If no header is given, an attempt is made to construct the header from
        these.  For a standard header, this would include the following.

    --- Header keywords : (see :meth:`~baseband.vdif.VDIFHeader.fromvalues`)

    time : `~astropy.time.Time`
        As an alternative, one can pass on ``ref_epoch`` and ``seconds``.
    nchan : int, optional
        Number of channels within stream (default 1).
        Note: that different # of channels per thread is not supported.
    complex_data : bool
        Whether data is complex.
    bps : int
        Bits per sample (or real, imaginary component).
    samples_per_frame : int
        Number of complete samples per frame.  As an alternative, use
        ``frame_length``, the number of long words for header plus payload.
        For some EDV, this number is fixed (e.g., ``frame_length=629`` for
        ``edv=3``, which corresponds to 20000 real 2-bit samples per frame).
    station : 2 characters
        Or unsigned 2-byte integer.
    edv : {`False`, 0, 1, 2, 3, 4, 0xab}
        Extended Data Version.
    """
    def __init__(self, raw, nthread=1, sample_rate=None, header=None,
                 squeeze=True, **kwargs):
        if header is None:
            if sample_rate is not None:
                kwargs['sample_rate'] = sample_rate
            header = VDIFHeader.fromvalues(**kwargs)
        # No frame sets yet exist, so generate a sample shape from values.
        super(VDIFStreamWriter, self).__init__(
            raw, header, None, nthread, sample_rate=sample_rate,
            squeeze=squeeze)
        # Set sample rate in the header, if it's possible, and not set already.
        try:
            header_sample_rate = self.header0.sample_rate
        except AttributeError:
            pass
        else:
            if header_sample_rate == 0:
                self.header0.sample_rate = self.sample_rate
            assert self.header0.sample_rate == self.sample_rate
        self._data = np.zeros(
            (self.samples_per_frame,) + self._unsliced_shape,
            np.complex64 if self.complex_data else np.float32)

    def write(self, data, invalid_data=False):
        """Write data, using multiple files as needed.

        Parameters
        ----------
        data : array
            Piece of data to be written, with sample dimensions as given by
            ``sample_shape``. This should be properly scaled to make best use
            of the dynamic range delivered by the encoding.
        invalid_data : bool, optional
            Whether the current data is valid.  Defaults to `False`.
        """
        assert data.shape[1:] == self.sample_shape, (
            "'data' should have trailing shape {}".format(self.sample_shape))
        assert data.dtype.kind == self._data.dtype.kind, (
            "'data' should be {}".format('complex' if self.data.dtype == 'c'
                                         else 'float'))
        if self.squeeze:
            data = self._unsqueeze(data)

        count = data.shape[0]
        sample = 0
        offset0 = self.offset
        frame = self._data
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
                self.write_frameset(self._data, self._header)

            self.offset += nsample
            count -= nsample


open = make_opener('VDIF', globals(), doc="""
--- For reading : (see :class:`VDIFStreamReader`)

subset : indexing object or tuple of objects, optional
    Specific components of the complete sample to decode.  If a single indexing
    object is passed, it selects threads.  If a tuple of objects is passed, the
    first selects threads and the second selects channels.  By default, all
    components are read.
sample_rate : `~astropy.units.Quantity`, optional
    Number of complete samples per second (ie. the rate at which each channel
    in each thread is sampled).  If not given, will be inferred from the header
    or by scanning one second of the file.
fill_value : float or complex
    Value to use for invalid or missing data. Default: 0.
squeeze : bool, optional
    If `True` (default), remove any dimensions of length unity from
    decoded data.

--- For writing : (see :class:`VDIFStreamWriter`)

nthread : int
    Number of threads the VLBI data has (e.g., 2 for 2 polarizations).
sample_rate : `~astropy.units.Quantity`, optional
    Number of complete samples per second (ie. the rate at which each
    channel in each thread is sampled).
squeeze : bool, optional
    If `True` (default), ``write`` accepts squeezed arrays as input,
    and adds channel and thread dimensions if they have length unity.
header : `~baseband.vdif.VDIFHeader`, optional
    Header for the first frame, holding time information, etc.
**kwargs
    If the header is not given, an attempt will be made to construct one
    with any further keyword arguments.  See :class:`VDIFStreamWriter`.

Returns
-------
Filehandle
    :class:`~baseband.vdif.base.VDIFFileReader` or
    :class:`~baseband.vdif.base.VDIFFileWriter` instance (binary), or
    :class:`~baseband.vdif.base.VDIFStreamReader` or
    :class:`~baseband.vdif.base.VDIFStreamWriter` instance (stream).
""")
