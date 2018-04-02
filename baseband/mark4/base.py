# Licensed under the GPLv3 - see LICENSE
from __future__ import division, unicode_literals, print_function

import numpy as np
from astropy.utils import lazyproperty
import astropy.units as u

from ..vlbi_base.base import (make_opener, VLBIFileBase, VLBIStreamBase,
                              VLBIStreamReaderBase, VLBIStreamWriterBase)
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

    Adds ``read_frame`` and ``locate_frame`` methods to the VLBI file wrapper.

    Parameters
    ----------
    ntrack : int or None, optional.
        Number of Mark 4 bitstreams.  Can be determined automatically as
        part of locating the first frame.
    decade : int or None
        Decade in which the observations were taken.  Can instead pass an
        approximate `ref_time`.
    ref_time : `~astropy.time.Time` or None
        Reference time within 4 years of the observation time.  Used only
        if `decade` is not given.
    """
    def __init__(self, fh_raw, ntrack=None, decade=None, ref_time=None):
        self.ntrack = ntrack
        self.decade = decade
        self.ref_time = ref_time
        super(Mark4FileReader, self).__init__(fh_raw)

    def read_frame(self):
        """Read a single frame (header plus payload).

        Returns
        -------
        frame : `~baseband.mark4.Mark4Frame`
            With ``.header`` and ``.data`` properties that return the
            :class:`~baseband.mark4.Mark4Header` and data encoded in the frame,
            respectively.
        """
        return Mark4Frame.fromfile(self.fh_raw, self.ntrack,
                                   decade=self.decade, ref_time=self.ref_time)

    def locate_frame(self, forward=True, maximum=None):
        """Locate the frame nearest the current position.

        The search is for the following pattern:

        * 32*tracks bits set at offset bytes
        * 1*tracks bits unset before offset
        * 32*tracks bits set at offset+2500*tracks bytes

        This reflects 'sync_pattern' of 0xffffffff for a given header and one
        a frame ahead, which is in word 2, plus the lsb of word 1, which is
        'system_id'.

        If the file does not have ntrack is set, it will be auto-determined.

        Parameters
        ----------
        forward : bool, optional
            Whether to search forwards or backwards.  Default: `True`.
        maximum : int, optional
            Maximum number of bytes forward to search through.
            Default: twice the framesize (of 20000 * ntrack // 8).

        Returns
        -------
        offset : int or `None`
            Byte offset of the next frame. `None` if the search was not
            successful.
        """
        fh = self.fh_raw
        file_pos = fh.tell()
        # Use initializer value (determines ntrack if not already given).
        ntrack = self.ntrack
        if ntrack is None:
            fh.seek(0)
            ntrack = self.determine_ntrack(maximum=maximum)
            if ntrack is None:
                raise ValueError("cannot determine ntrack automatically. "
                                 "Try passing in an explicit value.")
            if forward and fh.tell() >= file_pos:
                return fh.tell()

            fh.seek(file_pos)

        nset = np.ones(32 * ntrack // 8, dtype=np.int16)
        nunset = np.ones(ntrack // 8, dtype=np.int16)
        framesize = ntrack * 2500
        fh.seek(0, 2)
        filesize = fh.tell()
        if filesize < framesize:
            fh.seek(file_pos)
            return None

        if maximum is None:
            maximum = 2 * framesize
        # Loop over chunks to try to find the frame marker.
        step = framesize // 2
        # Read a bit more at every step to ensure we don't miss a "split"
        # header.
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
            # Check candidates by seeing whether there is a sync word
            # a framesize ahead. (Note: loop can be empty.)
            for possibility in possibilities[::1 if forward else -1]:
                # Real start of possible header.
                frame_start = frame + possibility - 63 * ntrack // 8
                if (forward and frame_start < file_pos or
                        not forward and frame_start > file_pos):
                    continue
                # Check there is a header following this.
                check = frame_start + framesize
                if check >= filesize - 32 * 2 * ntrack // 8 - len(nunset):
                    # But do before this one if we're beyond end of file.
                    check = frame_start - framesize
                    if check < 0:  # Assume OK if only one frame fits in file.
                        if frame_start + framesize > filesize:
                            continue
                        else:
                            break

                fh.seek(check + 32 * 2 * ntrack // 8)
                check_data = np.frombuffer(fh.read(len(nunset)),
                                           dtype=np.uint8)
                databits2 = nbits[check_data]
                if np.all(databits2 >= 6):
                    break  # Got it!

            else:  # None of them worked, so do next block.
                continue

            fh.seek(frame_start)
            return frame_start

        fh.seek(file_pos)
        return None

    def determine_ntrack(self, maximum=None):
        """Determines the number of tracks, by seeking the next frame.

        Uses ``find_frame`` to look for the first occurrence of a frame from
        the current position for all supported ``ntrack`` values.  Returns the
        first ``ntrack`` for which ``find_frame`` is successful, setting
        the file's ``ntrack`` property appropriately, and leaving the
        file pointer at the start of the frame.

        Parameters
        ----------
        maximum : int, optional
            Maximum number of bytes forward to search through.
            Default: twice the framesize (of 20000 * ntrack // 8).

        Returns
        -------
        ntrack : int or None
            Number of Mark 4 bitstreams.  `None` if no frame was found.
        """
        # Currently only 16, 32 and 64-track frames supported.
        old_ntrack = self.ntrack
        for ntrack in 16, 32, 64:
            try:
                self.ntrack = ntrack
                if self.locate_frame(maximum=maximum) is not None:
                    return ntrack
            except Exception:
                self.ntrack = old_ntrack
                raise

        self.ntrack = old_ntrack
        return None

    def find_header(self, forward=True, maximum=None):
        """Read header at the frame nearest the current position.

        The file pointer is left at the start of the header.

        Parameters
        ----------
        forward : bool, optional
            Seek forward if `True` (default), backward if `False`.
        maximum : int, optional
            Maximum number of bytes forward to search through.
            Default: twice the framesize (of 20000 * ntrack // 8).

        Returns
        -------
        header : :class:`~baseband.mark4.Mark4Header` or None
            Retrieved Mark 4 header, or `None` if nothing found.
        """
        offset = self.locate_frame(forward=forward)
        if offset is None:
            return None
        header = Mark4Header.fromfile(self.fh_raw, ntrack=self.ntrack,
                                      decade=self.decade,
                                      ref_time=self.ref_time)
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
        data : `~numpy.ndarray` or `~baseband.mark4.Mark4Frame`
            If an array, a header should be given, which will be used to
            get the information needed to encode the array, and to construct
            the Mark 4 frame.
        header : `~baseband.mark4.Mark4Header`
            Can instead give keyword arguments to construct a header.  Ignored
            if payload is a :class:`~baseband.mark4.Mark4Frame` instance.
        **kwargs :
            If `header` is not given, these are used to initialize one.
        """
        if not isinstance(data, Mark4Frame):
            data = Mark4Frame.fromdata(data, header, **kwargs)
        return data.tofile(self.fh_raw)


class Mark4StreamBase(VLBIStreamBase):
    """Base for Mark 4 streams."""

    def __init__(self, fh_raw, header0, sample_rate=None, squeeze=True,
                 subset=(), fill_value=0.):
        super(Mark4StreamBase, self).__init__(
            fh_raw, header0=header0, sample_rate=sample_rate,
            samples_per_frame=header0.samples_per_frame,
            unsliced_shape=(header0.nchan,),
            bps=header0.bps, complex_data=False, squeeze=squeeze,
            subset=subset, fill_value=fill_value)
        self._framerate = int(round((self.sample_rate /
                                     self.samples_per_frame).to_value(u.Hz)))


class Mark4StreamReader(Mark4StreamBase, VLBIStreamReaderBase):
    """VLBI Mark 4 format reader.

    Allows access to a Mark 4 file as a continuous series of samples.  Parts
    of the data stream replaced by header values are filled in.

    Parameters
    ----------
    fh_raw : filehandle
        Filehandle of the raw Mark 4 stream.
    sample_rate : `~astropy.units.Quantity`, optional
        Number of complete samples per second, i.e. the rate at which each
        channel is sampled.  If `None`, will be inferred from scanning two
        frames of the file.
    ntrack : int or None, optional
        Number of Mark 4 bitstreams.  If `None` (default), will attempt to
        automatically detect it by scanning the file.
    decade : int or None
        Decade of the observation start time (eg. ``2010`` for 2018), needed to
        remove ambiguity in the Mark 4 time stamp.  Can instead pass an
        approximate `ref_time`.
    ref_time : `~astropy.time.Time` or None
        Reference time within 4 years of the start time of the observations.
        Used only if `decade` is not given.
    squeeze : bool, optional
        If `True` (default), remove any dimensions of length unity from
        decoded data.
    subset : indexing object, optional
        Specific channels of the complete sample to decode (after possible
        squeezing).  If an empty tuple (default), all channels are read.
    fill_value : float or complex, optional
        Value to use for invalid or missing data. Default: 0.
    """

    _sample_shape_maker = Mark4Payload._sample_shape_maker

    def __init__(self, fh_raw, sample_rate=None, ntrack=None, decade=None,
                 ref_time=None, squeeze=True, subset=(), fill_value=0.):

        if decade is None and ref_time is None:
            raise ValueError("Mark 4 stream reader requires decade or "
                             "ref_time. Please pass either explicitly.")

        # Get binary file reader.
        fh_raw = Mark4FileReader(fh_raw, ntrack=ntrack, decade=decade,
                                 ref_time=ref_time)
        # Find first header, determining ntrack if needed.
        header0 = fh_raw.find_header()
        assert header0 is not None, (
            "could not find a first frame using ntrack={}. Perhaps "
            "try ntrack=None for auto-determination.".format(ntrack))
        self._offset0 = fh_raw.tell()
        super(Mark4StreamReader, self).__init__(
            fh_raw, header0=header0, sample_rate=sample_rate,
            squeeze=squeeze, subset=subset, fill_value=fill_value)
        # Use reference time in preference to decade so that a stream wrapping
        # a decade will work.
        self.fh_raw.decade = None
        self.fh_raw.ref_time = self.start_time

    @staticmethod
    def _get_frame_rate(fh, header_template):
        """Returns the number of frames per second in a Mark 4 file.

        Parameters
        ----------
        fh : filehandle
            Filehandle of the raw Mark 4 stream.
        header_template : header class or instance
            Definition or instance of file format's header class.

        Returns
        -------
        framerate : `~astropy.units.Quantity`
            Frames per second.

        Notes
        -----
        Unlike `VLBIStreamReaderBase._get_frame_rate`, this function reads
        only two consecutive frames, extracting their timestamps to determine
        how much time has elapsed.  It will return an `EOFError` if there is
        only one frame.
        """
        oldpos = fh.tell()
        header0 = header_template.fromfile(fh, header_template.ntrack,
                                           ref_time=header_template.time)
        fh.seek(header0.payloadsize, 1)
        header1 = header_template.fromfile(fh, header_template.ntrack,
                                           ref_time=header_template.time)
        fh.seek(oldpos)
        # Mark 4 specification states frames-lengths range from 1.25 ms
        # to 160 ms.
        tdelta = header1.ms[0] - header0.ms[0]
        return np.round(1000. / tdelta) * u.Hz

    @lazyproperty
    def _last_header(self):
        """Last header of the file."""
        last_header = super(Mark4StreamReader, self)._last_header
        # Infer the decade, assuming the end of the file is no more than
        # 4 years away from the start.
        last_header.infer_decade(self.start_time)
        return last_header

    def _read_frame(self, index):
        self.fh_raw.seek(self._offset0 + index * self.header0.framesize)
        frame = self.fh_raw.read_frame()
        # Set decoded value for invalid data.
        frame.invalid_data_value = self.fill_value
        # TODO: add check that we got the right frame.
        return frame


class Mark4StreamWriter(Mark4StreamBase, VLBIStreamWriterBase):
    """VLBI Mark 4 format writer.

    Encodes and writes sequences of samples to file.

    Parameters
    ----------
    raw : filehandle
        Which will write filled sets of frames to storage.
    header0 : `~baseband.mark4.Mark4Header`
        Header for the first frame, holding time information, etc.  Can instead
        give keyword arguments to construct a header (see ``**kwargs``).
    sample_rate : `~astropy.units.Quantity`
        Number of complete samples per second, i.e. the rate at which each
        channel is sampled.  Needed to calculate header timestamps.
    squeeze : bool, optional
        If `True` (default), ``write`` accepts squeezed arrays as input, and
        adds any dimensions of length unity.
    **kwargs
        If no header is given, an attempt is made to construct one from these.
        For a standard header, this would include the following.

    --- Header keywords : (see :meth:`~baseband.mark4.Mark4Header.fromvalues`)

    time : `~astropy.time.Time`
        Start time of the file.  Sets bcd-encoded unit year, day, hour, minute,
        second in the header.
    ntrack : int
        Number of Mark 4 bitstreams (equal to number of channels times
        ``fanout`` times ``bps``)
    bps : int
        Bits per elementary sample.
    fanout : int
        Number of tracks over which a given channel is spread out.
    """

    _sample_shape_maker = Mark4Payload._sample_shape_maker

    def __init__(self, fh_raw, header0=None, sample_rate=None, squeeze=True,
                 **kwargs):
        if header0 is None:
            header0 = Mark4Header.fromvalues(**kwargs)
        super(Mark4StreamWriter, self).__init__(
            fh_raw=fh_raw, header0=header0, sample_rate=sample_rate,
            squeeze=squeeze)
        # Set up initial payload with right shape.
        samples_per_payload = (
            header0.samples_per_frame * header0.payloadsize //
            header0.framesize)
        self._payload = Mark4Payload.fromdata(
            np.zeros((samples_per_payload, header0.nchan), np.float32),
            header0)

    def _make_frame(self, frame_index):
        header = self.header0.copy()
        header.update(time=self.start_time + frame_index /
                      self._framerate * u.s)
        # Reuse payload.
        return Mark4Frame(header, self._payload)


open = make_opener('Mark4', globals(), doc="""
--- For reading a stream : (see `~baseband.mark4.base.Mark4StreamReader`)

sample_rate : `~astropy.units.Quantity`, optional
    Number of complete samples per second, i.e. the rate at which each channel
    is sampled.  If not given, will be inferred from scanning two frames of
    the file.
ntrack : int, optional
    Number of Mark 4 bitstreams.  If `None` (default), will attempt to
    automatically detect it by scanning the file.
decade : int or None
    Decade of the observation start time (eg. ``2010`` for 2018), needed to
    remove ambiguity in the Mark 4 time stamp (default: `None`).  Can instead
    pass an approximate `ref_time`.
ref_time : `~astropy.time.Time` or None
    Reference time within 4 years of the start time of the observations.  Used
    only if `decade` is not given.
squeeze : bool, optional
    If `True` (default), remove any dimensions of length unity from
    decoded data.
subset : indexing object, optional
    Specific channels of the complete sample to decode (after possible
    squeezing).  If an empty tuple (default), all channels are read.
fill_value : float or complex, optional
    Value to use for invalid or missing data. Default: 0.

--- For writing a stream : (see `~baseband.mark4.base.Mark4StreamWriter`)

header0 : `~baseband.mark4.Mark4Header`
    Header for the first frame, holding time information, etc.  Can instead
    give keyword arguments to construct a header (see ``**kwargs``).
sample_rate : `~astropy.units.Quantity`
    Number of complete samples per second, i.e. the rate at which each channel
    is sampled.  Needed to calculate header timestamps.
squeeze : bool, optional
    If `True` (default), ``write`` accepts squeezed arrays as input, and adds
    any dimensions of length unity.
**kwargs
    If the header is not given, an attempt will be made to construct one
    with any further keyword arguments.  See
    :class:`~baseband.mark4.base.Mark4StreamWriter`.

Returns
-------
Filehandle
    :class:`~baseband.mark4.base.Mark4FileReader` or
    :class:`~baseband.mark4.base.Mark4FileWriter` (binary), or
    :class:`~baseband.mark4.base.Mark4StreamReader` or
    :class:`~baseband.mark4.base.Mark4StreamWriter` (stream)
""")
