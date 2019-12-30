# Licensed under the GPLv3 - see LICENSE
import numpy as np
from astropy.utils import lazyproperty, deprecated
import astropy.units as u

from ..vlbi_base.base import (make_opener, VLBIFileBase, VLBIFileReaderBase,
                              VLBIStreamBase, VLBIStreamReaderBase,
                              VLBIStreamWriterBase, HeaderNotFoundError)
from .header import Mark4Header
from .payload import Mark4Payload
from .frame import Mark4Frame
from .file_info import Mark4FileReaderInfo


__all__ = ['Mark4FileReader', 'Mark4FileWriter',
           'Mark4StreamBase', 'Mark4StreamReader', 'Mark4StreamWriter',
           'open']

# Look-up table for the number of bits in a byte.
nbits = ((np.arange(256)[:, np.newaxis] >> np.arange(8) & 1)
         .sum(1).astype(np.int16))


class Mark4FileReader(VLBIFileReaderBase):
    """Simple reader for Mark 4 files.

    Wraps a binary filehandle, providing methods to help interpret the data,
    such as `locate_frame`, `read_frame` and `get_frame_rate`.

    Parameters
    ----------
    fh_raw : filehandle
        Filehandle of the raw binary data file.
    ntrack : int or None, optional.
        Number of Mark 4 bitstreams.  Can be determined automatically as
        part of locating the first frame.
    decade : int or None
        Decade in which the observations were taken.  Can instead pass an
        approximate ``ref_time``.
    ref_time : `~astropy.time.Time` or None
        Reference time within 4 years of the observation time.  Used only
        if ``decade`` is not given.
    """

    def __init__(self, fh_raw, ntrack=None, decade=None, ref_time=None):
        self.ntrack = ntrack
        self.decade = decade
        self.ref_time = ref_time
        super().__init__(fh_raw)

    def __repr__(self):
        return ("{name}(fh_raw={s.fh_raw}, ntrack={s.ntrack}, "
                "decade={s.decade}, ref_time={s.ref_time})"
                .format(name=self.__class__.__name__, s=self))

    info = Mark4FileReaderInfo()

    def read_header(self):
        """Read a single header from the file.

        Returns
        -------
        header : `~baseband.mark4.Mark4Header`
        """
        return Mark4Header.fromfile(self, ntrack=self.ntrack,
                                    decade=self.decade, ref_time=self.ref_time)

    def read_frame(self, verify=True):
        """Read a single frame (header plus payload).

        Returns
        -------
        frame : `~baseband.mark4.Mark4Frame`
            With ``.header`` and ``.data`` properties that return the
            :class:`~baseband.mark4.Mark4Header` and data encoded in the frame,
            respectively.
        verify : bool, optional
            Whether to do basic checks of frame integrity.  Default: `True`.
        """
        return Mark4Frame.fromfile(self.fh_raw, self.ntrack,
                                   decade=self.decade, ref_time=self.ref_time,
                                   verify=verify)

    def get_frame_rate(self):
        """Determine the number of frames per second.

        The frame rate is calculated from the time elapsed between the
        first two frames, as inferred from their time stamps.

        Returns
        -------
        frame_rate : `~astropy.units.Quantity`
            Frames per second.
        """
        with self.temporary_offset():
            self.seek(0)
            header0 = self.find_header()
            self.seek(header0.frame_nbytes, 1)
            header1 = self.read_header()

        # Mark 4 specification states frames-lengths range from 1.25 ms
        # to 160 ms.
        tdelta = (header1.fraction[0] - header0.fraction[0]) % 1.
        return u.Quantity(1 / tdelta, u.Hz).round()

    def locate_frames(self, pattern=None, *, mask=None, frame_nbytes=None,
                      offset=0, forward=True, maximum=None, check=1):
        """Use a pattern to locate frame starts near the current position.

        Parameters
        ----------
        pattern : header, ~numpy.ndaray, bytes, or (iterable of) int, optional
            Synchronization pattern to look for.  The default uses the
            Mark 4 sync pattern, plus that the bit before is 0. See
            `~baseband.mark4.header.Mark4Header.invariant_pattern`.
        mask : ~numpy.ndarray, bytes, int, or iterable of int.
            Bit mask for the pattern, with 1 indicating a given bit will
            be used the comparison.  Only used if ``pattern`` is given
            and is not a header.
        frame_nbytes : int, optional
            Frame size in bytes.  Defaults to the frame size for
            ``ntrack``.  If given, overrides ``self.ntrack``.
        offset : int, optional
            Offset from the frame start that the pattern occurs.  Only
            used if ``pattern`` is given and not a header.
        forward : bool, optional
            Seek forward if `True` (default), backward if `False`.
        maximum : int, optional
            Maximum number of bytes to search away from the present location.
            Use 0 to check only at the current position.
        check : int or tuple of int, optional
            Frame offsets where another sync pattern should be present (if
            inside the file). Default: 1, i.e., a sync pattern should be
            present one frame after the one found (independent of
            ``forward``), thus helping to guarantee the frame is OK.

        Returns
        -------
        locations : list of int
            Locations of sync patterns within the range scanned,
            in order of proximity to the starting position.
        """
        # Use initializer value (determines ntrack if not already given).
        if frame_nbytes is None:
            ntrack = self.ntrack
            if ntrack is None:
                with self.temporary_offset():
                    self.seek(0)
                    ntrack = self.determine_ntrack(maximum=maximum)

            frame_nbytes = ntrack * 2500

        else:
            ntrack, resid = divmod(frame_nbytes, 2500)
            if resid:
                raise ValueError('frame_nbytes must be a multiple of '
                                 '2500 bytes for Mark 4 data.')

        if pattern is None:
            pattern, mask = Mark4Header.invariant_pattern(ntrack=self.ntrack)
        return super().locate_frames(
            pattern, mask=mask, frame_nbytes=frame_nbytes, offset=offset,
            forward=forward, maximum=maximum, check=check)

    def determine_ntrack(self, maximum=None):
        """Determines the number of tracks, by seeking the next frame.

        Uses `locate_frame` to look for the first occurrence of a frame from
        the current position for all supported ``ntrack`` values.  Returns the
        first ``ntrack`` for which `locate_frame` is successful, setting
        the file's ``ntrack`` property appropriately, and leaving the
        file pointer at the start of the frame.

        Parameters
        ----------
        maximum : int, optional
            Maximum number of bytes forward to search through.
            Default: twice the frame size (``20000 * ntrack // 8``).

        Returns
        -------
        ntrack : int or None
            Number of Mark 4 bitstreams.

        Raises
        ------
        ~baseband.vlbi_base.base.HeaderNotFoundError
            If no frame was found for any value of ntrack.
        """
        # Currently only 16, 32 and 64-track frames supported.
        old_ntrack = self.ntrack
        trials = 16, 32, 64
        for ntrack in trials:
            self.ntrack = ntrack
            try:
                self.find_header(maximum=maximum)
                return ntrack
            except Exception:
                pass

        self.ntrack = old_ntrack
        raise HeaderNotFoundError("cannot determine ntrack automatically. "
                                  "(tried {}). Try passing in an "
                                  "explicit value.".format(trials))

    @deprecated(since='3.1', alternative='locate_frames or find_header')
    def locate_frame(self, *args, **kwargs):
        """Use a pattern to locate the frame nearest the current position.

        Like ``locate_frames``, but selects the closest frame and leaves
        the file pointer at its position.

        Returns
        -------
        location : int
            The location of the file pointer.

        Raises
        ------
        ~baseband.vlbi_base.base.HeaderNotFoundError
            If no frame was found.
        """
        locations = self.locate_frames(*args, **kwargs)
        if not locations:
            raise HeaderNotFoundError('could not locate a a nearby frame.')

        return self.seek(locations[0])


class Mark4FileWriter(VLBIFileBase):
    """Simple writer for Mark 4 files.

    Adds `write_frame` method to the VLBI binary file wrapper.
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
            If ``header`` is not given, these are used to initialize one.
        """
        if not isinstance(data, Mark4Frame):
            data = Mark4Frame.fromdata(data, header, **kwargs)
        return data.tofile(self.fh_raw)


class Mark4StreamBase(VLBIStreamBase):
    """Base for Mark 4 streams."""

    def __init__(self, fh_raw, header0, sample_rate=None, squeeze=True,
                 subset=(), fill_value=0., verify=True):
        super().__init__(
            fh_raw, header0=header0, sample_rate=sample_rate,
            samples_per_frame=header0.samples_per_frame,
            unsliced_shape=(header0.nchan,),
            bps=header0.bps, complex_data=False, squeeze=squeeze,
            subset=subset, fill_value=fill_value, verify=verify)


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
        approximate ``ref_time``.
    ref_time : `~astropy.time.Time` or None
        Reference time within 4 years of the start time of the observations.
        Used only if ``decade`` is not given.
    squeeze : bool, optional
        If `True` (default), remove any dimensions of length unity from
        decoded data.
    subset : indexing object, optional
        Specific channels of the complete sample to decode (after possible
        squeezing).  If an empty tuple (default), all channels are read.
    fill_value : float or complex, optional
        Value to use for invalid or missing data. Default: 0.
    verify : bool or str, optional
        Whether to do basic checks of frame integrity when reading.
        Default: 'fix', which implies basic verification and replacement
        of gaps with zeros.
    """

    _sample_shape_maker = Mark4Payload._sample_shape_maker

    def __init__(self, fh_raw, sample_rate=None, ntrack=None, decade=None,
                 ref_time=None, squeeze=True, subset=(), fill_value=0.,
                 verify='fix'):

        if decade is None and ref_time is None:
            raise TypeError("Mark 4 stream reader requires either decade or "
                            "ref_time to be passed in.")

        # Get binary file reader.
        fh_raw = Mark4FileReader(fh_raw, ntrack=ntrack, decade=decade,
                                 ref_time=ref_time)
        # Find first header, determining ntrack if needed.
        try:
            header0 = fh_raw.find_header()
        except Exception as exc:
            if ntrack is not None:
                exc.args += ("could not find a first frame using ntrack={}. "
                             "Perhaps try ntrack=None for auto-determination."
                             .format(ntrack),)
            raise exc

        super().__init__(
            fh_raw, header0=header0, sample_rate=sample_rate,
            squeeze=squeeze, subset=subset, fill_value=fill_value,
            verify=verify)
        self._raw_offsets[0] = fh_raw.tell()
        # Use reference time in preference to decade so that a stream wrapping
        # a decade will work.
        self.fh_raw.decade = None
        self.fh_raw.ref_time = self.start_time

    @lazyproperty
    def _last_header(self):
        """Last header of the file."""
        last_header = super()._last_header
        # Infer the decade, assuming the end of the file is no more than
        # 4 years away from the start.
        last_header.infer_decade(self.start_time)
        return last_header


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
        If `True` (default), `write` accepts squeezed arrays as input, and
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
        super().__init__(fh_raw=fh_raw, header0=header0,
                         sample_rate=sample_rate, squeeze=squeeze)
        # Set up initial payload with right shape.
        samples_per_payload = (
            header0.samples_per_frame * header0.payload_nbytes
            // header0.frame_nbytes)
        self._payload = Mark4Payload.fromdata(
            np.zeros((samples_per_payload, header0.nchan), np.float32),
            header0)

    def _make_frame(self, frame_index):
        header = self.header0.copy()
        header.update(time=self.start_time + frame_index
                      / self._frame_rate)
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
    pass an approximate ``ref_time``.
ref_time : `~astropy.time.Time` or None
    Reference time within 4 years of the start time of the observations.  Used
    only if ``decade`` is not given.
squeeze : bool, optional
    If `True` (default), remove any dimensions of length unity from
    decoded data.
subset : indexing object, optional
    Specific channels of the complete sample to decode (after possible
    squeezing).  If an empty tuple (default), all channels are read.
fill_value : float or complex, optional
    Value to use for invalid or missing data. Default: 0.
verify : bool or 'fix', optional
    Whether to do basic checks of frame integrity when reading.
    Default: 'fix', which implies basic verification and replacement
    of gaps with zeros.

--- For writing a stream : (see `~baseband.mark4.base.Mark4StreamWriter`)

header0 : `~baseband.mark4.Mark4Header`
    Header for the first frame, holding time information, etc.  Can instead
    give keyword arguments to construct a header (see ``**kwargs``).
sample_rate : `~astropy.units.Quantity`
    Number of complete samples per second, i.e. the rate at which each channel
    is sampled.  Needed to calculate header timestamps.
squeeze : bool, optional
    If `True` (default), writer accepts squeezed arrays as input, and adds
    any dimensions of length unity.
file_size : int or None, optional
    When writing to a sequence of files, the maximum size of one file in bytes.
    If `None` (default), the file size is unlimited, and only the first
    file will be written to.
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

Notes
-----
Although it is not generally expected to be useful for Mark 4, like for
other formats one can also pass to ``name`` a list, tuple, or subclass of
`~baseband.helpers.sequentialfile.FileNameSequencer`.  For writing to multiple
files, the ``file_size`` keyword must be passed or only the first file will be
written to.  One may also pass in a `~baseband.helpers.sequentialfile` object
(opened in 'rb' mode for reading or 'w+b' for writing), though for typical use
cases it is practically identical to passing in a list or template.
""")
