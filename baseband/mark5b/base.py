# Licensed under the GPLv3 - see LICENSE
import numpy as np
import astropy.units as u
from astropy.utils import lazyproperty

from ..base.base import (
    FileBase, VLBIFileReaderBase,
    VLBIStreamReaderBase, StreamWriterBase,
    FileOpener, FileInfo)
from .header import Mark5BHeader
from .payload import Mark5BPayload
from .frame import Mark5BFrame
from .file_info import Mark5BFileReaderInfo


__all__ = ['Mark5BFileReader', 'Mark5BFileWriter',
           'Mark5BStreamBase', 'Mark5BStreamReader', 'Mark5BStreamWriter',
           'open', 'info']


class Mark5BFileReader(VLBIFileReaderBase):
    """Simple reader for Mark 5B files.

    Wraps a binary filehandle, providing methods to help interpret the data,
    such as `read_frame` and `get_frame_rate`.

    Parameters
    ----------
    fh_raw : filehandle
        Filehandle of the raw binary data file.
    kday : int or None
        Explicit thousands of MJD of the observation time.  Can instead
        pass an approximate ``ref_time``.
    ref_time : `~astropy.time.Time` or None
        Reference time within 500 days of the observation time, used to
        infer the full MJD.  Used only if ``kday`` is not given.
    nchan : int, optional
        Number of channels.   Default: 1.
    bps : int, optional
        Bits per elementary sample.  Default: 2.
    """

    def __init__(self, fh_raw, kday=None, ref_time=None, nchan=None, bps=2):
        self.kday = kday
        self.ref_time = ref_time
        self.nchan = nchan
        self.bps = bps
        super().__init__(fh_raw)

    def __repr__(self):
        return ("{name}(fh_raw={s.fh_raw}, kday={s.kday}, "
                "ref_time={s.ref_time}, nchan={s.nchan}, bps={s.bps})"
                .format(name=self.__class__.__name__, s=self))

    info = Mark5BFileReaderInfo()

    def read_header(self):
        """Read a single header from the file.

        Returns
        -------
        header : `~baseband.mark5b.Mark5BHeader`
        """
        return Mark5BHeader.fromfile(self, kday=self.kday,
                                     ref_time=self.ref_time)

    def read_frame(self, verify=True):
        """Read a single frame (header plus payload).

        Returns
        -------
        frame : `~baseband.mark5b.Mark5BFrame`
            With ``header`` and ``data`` properties that return the
            `~baseband.mark5b.Mark5BHeader` and data encoded in the frame,
            respectively.
        verify : bool, optional
            Whether to do basic checks of frame integrity.  Default: `True`.
        """
        if self.nchan is None:
            raise TypeError("In order to read frames, the file handle should "
                            "be initialized with nchan set.")
        return Mark5BFrame.fromfile(self.fh_raw, kday=self.kday,
                                    ref_time=self.ref_time,
                                    sample_shape=(self.nchan,),
                                    bps=self.bps, verify=verify)

    def get_frame_rate(self):
        """Determine the number of frames per second.

        This method first tries to determine the frame rate by looking for
        the highest frame number in the first second of data.  If that fails,
        it uses the time difference between two consecutive frames. This can
        fail if the headers do not store fractional seconds, or if the data
        rate is above 512 Mbps.

        Returns
        -------
        frame_rate : `~astropy.units.Quantity`
            Frames per second.
        """
        try:
            return super().get_frame_rate()
        except Exception as exc:
            with self.temporary_offset(0):
                try:
                    header0 = self.read_header()
                    self.seek(header0.payload_nbytes, 1)
                    header1 = self.read_header()
                    tdelta = header1.fraction - header0.fraction
                    if tdelta == 0.:
                        exc.args += ("frame rate can also not be determined "
                                     "from the first two headers, as they "
                                     "have identical fractional seconds.",)
                    return u.Quantity(1 / tdelta, u.Hz).round()
                except Exception:
                    pass
            raise exc

    def locate_frames(self, pattern=None, **kwargs):
        """Use a pattern to locate frame starts near the current position.

        Note that the current position is always included.

        Parameters are as for
        `baseband.base.base.VLBIFileReaderBase.locate_frames`
        except that by default the Mark 5B sync pattern is used.
        """
        if pattern is None:
            pattern = Mark5BHeader
        return super().locate_frames(pattern, **kwargs)


class Mark5BFileWriter(FileBase):
    """Simple writer for Mark 5B files.

    Adds `write_frame` method to the VLBI binary file wrapper.
    """

    def write_frame(self, data, header=None, bps=2, valid=True, **kwargs):
        """Write a single frame (header plus payload).

        Parameters
        ----------
        data : `~numpy.ndarray` or :`~baseband.mark5b.Mark5BFrame`
            If an array, ``header`` should be given, which will be used to
            get the information needed to encode the array, and to construct
            the Mark 5B frame.
        header : `~baseband.mark5b.Mark5BHeader`
            Can instead give keyword arguments to construct a header.  Ignored
            if ``data`` is a `~baseband.mark5b.Mark5BFrame` instance.
        bps : int, optional
            Bits per elementary sample, to use when encoding the payload.
            Ignored if ``data`` is a `~baseband.mark5b.Mark5BFrame` instance.
            Default: 2.
        valid : bool, optional
            Whether the data are valid; if `False`, a payload filled with an
            appropriate pattern will be crated.  Ignored if ``data`` is a
            `~baseband.mark5b.Mark5BFrame` instance.  Default: `True`.
        **kwargs
            If ``header`` is not given, these are used to initialize one.
        """
        if not isinstance(data, Mark5BFrame):
            data = Mark5BFrame.fromdata(data, header, bps=bps, valid=valid,
                                        **kwargs)
        return data.tofile(self.fh_raw)


class Mark5BStreamBase:
    """Provides sample shape maker and fast time and index getting/setting."""

    _sample_shape_maker = Mark5BPayload._sample_shape_maker

    def _get_time(self, header):
        return header.get_time(frame_rate=self._frame_rate)

    def _set_time(self, header, time):
        # Pass frame_rate, which is needed to calculate the frame number,
        # and use update so that also the CRC is updated.
        header.update(time=time, frame_rate=self._frame_rate)

    def _get_index(self, header):
        # Override to provide index faster, without calculating times.
        # TODO: OK to ignore leap seconds? Not sure what writer does.
        return int(round(self._frame_rate.to_value(u.Hz)
                         * (header.seconds - self.header0.seconds
                            + 86400 * (header.kday - self.header0.kday
                                       + header.jday - self.header0.jday))
                         + header['frame_nr'] - self.header0['frame_nr']))

    def _set_index(self, header, index):
        # Override to avoid explicit time calculations.
        dt, frame_nr = divmod(index + self.header0['frame_nr'],
                              int(round(self._frame_rate.to_value(u.Hz))))
        fraction = (frame_nr/self._frame_rate).to_value(u.s)
        dd, seconds = divmod(dt + self.header0.seconds, 86400)
        dk, jday = divmod(dd + self.header0.jday, 1000)
        kday = dk*1000 + self.header0.kday
        # Use update to ensure CRC is updated as well.
        header.update(frame_nr=frame_nr, fraction=fraction,
                      seconds=seconds, jday=jday, kday=kday)


class Mark5BStreamReader(Mark5BStreamBase, VLBIStreamReaderBase):
    """VLBI Mark 5B format reader.

    Allows access a Mark 5B file as a continues series of samples.

    Parameters
    ----------
    fh_raw : filehandle
        Filehandle of the raw Mark 5B stream.
    sample_rate : `~astropy.units.Quantity`, optional
        Number of complete samples per second, i.e. the rate at which each
        channel is sampled.  If `None` (default), will be inferred from
        scanning one second of the file or, failing that, using the time
        difference between two consecutive frames.
    kday : int or None
        Explicit thousands of MJD of the observation start time (eg. ``57000``
        for MJD 57999), used to infer the full MJD from the header's time
        information.  Can instead pass an approximate ``ref_time``.
    ref_time : `~astropy.time.Time` or None
        Reference time within 500 days of the observation start time, used
        to infer the full MJD.  Only used if ``kday`` is not given.
    nchan : int
        Number of channels.  Needs to be explicitly passed in.
    bps : int, optional
        Bits per elementary sample.  Default: 2.
    squeeze : bool, optional
        If `True` (default), remove any dimensions of length unity from
        decoded data.
    subset : indexing object, optional
        Specific channels of the complete sample to decode (after possible
        squeezing). If an empty tuple (default), all channels are read.
    fill_value : float or complex
        Value to use for invalid or missing data. Default: 0.
    verify : bool or 'fix', optional
        Whether to do basic checks of frame integrity when reading.
        Default: 'fix', which implies basic verification and replacement
        of gaps with zeros.
    """

    def __init__(self, fh_raw, sample_rate=None, kday=None, ref_time=None,
                 nchan=None, bps=2, squeeze=True, subset=(), fill_value=0.,
                 verify='fix'):

        if nchan is None:
            raise TypeError("Mark 5B stream reader requires nchan to be "
                            "explicity passed in.")

        if kday is None and ref_time is None:
            raise TypeError("Mark 5B stream reader requires either kday or "
                            "ref_time to be passed in.")

        fh_raw = Mark5BFileReader(fh_raw, nchan=nchan, bps=bps,
                                  ref_time=ref_time, kday=kday)
        header0 = fh_raw.find_header()
        super().__init__(
            fh_raw, header0, sample_rate=sample_rate,
            samples_per_frame=header0.payload_nbytes * 8 // bps // nchan,
            sample_shape=(nchan,), bps=bps,
            squeeze=squeeze, subset=subset, fill_value=fill_value,
            verify=verify)
        # Use ref_time in preference to kday so we can handle files that
        # span a change in 1000s of MJD.
        self.fh_raw.kday = None
        self.fh_raw.ref_time = self.start_time

    @lazyproperty
    def _last_header(self):
        """Last header of the file."""
        last_header = super()._last_header
        # Infer kday, assuming the end of the file is no more than
        # 500 days away from the start.
        last_header.infer_kday(self.start_time)
        return last_header


class Mark5BStreamWriter(Mark5BStreamBase, StreamWriterBase):
    """VLBI Mark 5B format writer.

    Encodes and writes sequences of samples to file.

    Parameters
    ----------
    fh_raw : filehandle
        For writing filled sets of frames to storage.
    header0 : `~baseband.mark5b.Mark5BHeader`
        Header for the first frame, holding time information, etc.
    sample_rate : `~astropy.units.Quantity`
        Number of complete samples per second, i.e. the rate at which each
        channel is sampled.  Needed to calculate header timestamps.
    nchan : int, optional
        Number of channels.  Default: 1.
    bps : int, optional
        Bits per elementary sample.  Default: 2.
    squeeze : bool, optional
        If `True` (default), `write` accepts squeezed arrays as input, and
        adds any dimensions of length unity.
    """

    def __init__(self, fh_raw, header0=None, sample_rate=None, nchan=1, bps=2,
                 squeeze=True):
        fh_raw = Mark5BFileWriter(fh_raw)
        super().__init__(
            fh_raw, header0, sample_rate=sample_rate,
            samples_per_frame=header0.payload_nbytes * 8 // bps // nchan,
            sample_shape=(nchan,), bps=bps, squeeze=squeeze)
        # Initial frame, reused for every other one.
        self._frame = Mark5BFrame.fromdata(
            np.zeros((self.samples_per_frame, nchan)),
            header0.copy(), bps=bps)


class Mark5BFileOpener(FileOpener):
    def get_header0(self, kwargs):
        if ('time' in kwargs and 'sample_rate' in kwargs
                and 'frame_rate' not in kwargs):
            bps = kwargs.get('bps', 2)
            nchan = kwargs.get('nchan', 1)
            samples_per_frame = Mark5BHeader.payload_nbytes * 8 // bps // nchan
            kwargs['frame_rate'] = kwargs['sample_rate'] / samples_per_frame
        header0 = super().get_header0(kwargs)
        kwargs.pop('frame_rate', None)
        return header0


open = Mark5BFileOpener.create(globals(), doc="""
--- For reading a stream : (see `~baseband.mark5b.base.Mark5BStreamReader`)

sample_rate : `~astropy.units.Quantity`, optional
    Number of complete samples per second, i.e. the rate at which each channel
    is sampled.  If `None` (default), will be inferred from scanning one
    second of the file or, failing that, using the time difference between two
    consecutive frames.
kday : int or None
    Explicit thousands of MJD of the observation start time (eg. ``57000`` for
    MJD 57999), used to infer the full MJD from the header's time information.
    Can instead pass an approximate ``ref_time``.
ref_time : `~astropy.time.Time` or None
    Reference time within 500 days of the observation start time, used to infer
    the full MJD.  Only used if ``kday`` is not given.
nchan : int, optional
    Number of channels.  Default: 1.
bps : int, optional
    Bits per elementary sample.  Default: 2.
squeeze : bool, optional
    If `True` (default), remove any dimensions of length unity from
    decoded data.
subset : indexing object, optional
    Specific channels of the complete sample to decode (after possible
    squeezing). If an empty tuple (default), all channels are read.
fill_value : float or complex
    Value to use for invalid or missing data. Default: 0.
verify : bool or 'fix', optional
    Whether to do basic checks of frame integrity when reading.
    Default: 'fix', which implies basic verification and replacement
    of gaps with zeros.

--- For writing a stream : (see `~baseband.mark5b.base.Mark5BStreamWriter`)

header0 : :class:`~baseband.mark5b.Mark5BHeader`
    Header for the first frame, holding time information, etc.  Can instead
    give keyword arguments to construct a header (see ``**kwargs``).
sample_rate : `~astropy.units.Quantity`
    Number of complete samples per second, i.e. the rate at which each
    channel is sampled.  Needed to calculate header timestamps.
nchan : int, optional
    Number of channels.  Default: 1.
bps : int, optional
    Bits per elementary sample.  Default: 2.
squeeze : bool, optional
    If `True` (default), writer accepts squeezed arrays as input,
    and adds channel and thread dimensions if they have length unity.
file_size : int or None, optional
    When writing to a sequence of files, the maximum size of one file in bytes.
    If `None` (default), the file size is unlimited, and only the first
    file will be written to.
**kwargs
    If no header is given, an attempt is made to construct one from these.
    For a standard header, the following suffices.

--- Header kwargs : (see :meth:`~baseband.mark5b.Mark5BHeader.fromvalues`)

time : `~astropy.time.Time`
    Start time of the file.  Sets bcd-encoded unit day, hour, minute,
    second, and fraction, as well as the frame number, in the header.

Returns
-------
Filehandle
    :class:`~baseband.mark5b.base.Mark5BFileReader` or
    :class:`~baseband.mark5b.base.Mark5BFileWriter` (binary), or
    :class:`~baseband.mark5b.base.Mark5BStreamReader` or
    :class:`~baseband.mark5b.base.Mark5BStreamWriter` (stream).

Notes
-----
One can also pass to ``name`` a list, tuple, or subclass of
`~baseband.helpers.sequentialfile.FileNameSequencer`.  For writing to multiple
files, the ``file_size`` keyword must be passed or only the first file will be
written to.  One may also pass in a `~baseband.helpers.sequentialfile` object
(opened in 'rb' mode for reading or 'w+b' for writing), though for typical use
cases it is practically identical to passing in a list or template.
""")


info = FileInfo.create(globals())
