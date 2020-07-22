# Licensed under the GPLv3 - see LICENSE
"""Common classes for accessing baseband data as binary files and streams.

For access as binary files, the main `~baseband.base.base.FileBase` class
provides a base to which simple methods such as ``read_frame`` and
``write_frame`` can be added. Unlike normal binary file readers, the base can
be pickled: it stores the name and re-opens the file if unpickled.  The
`~baseband.base.base.VLBIFileReaderBase` adds additional methods useful for
streams with small frames, to locate frames and determine the frame rate.

For access as streams, the `~baseband.base.base.StreamReaderBase` and
`~baseband.base.base.StreamWriterBase` classes allow one to process data in
terms of frames, and access them as streams of samples via the ``read`` and
``write`` methods, providing a number of private methods which can be
overridden as needed for specific data formats.

The `~baseband.base.base.FileOpener` and `~baseband.base.base.FileInfo`
classes are to help create the ``open`` and ``info`` functions that are
expected to exist for each format.
"""
import io
import functools
import inspect
import textwrap
import warnings
import operator
from collections import namedtuple
from contextlib import contextmanager

import numpy as np
from numpy.lib.stride_tricks import as_strided
import astropy.units as u
from astropy.utils import lazyproperty

from ..helpers import sequentialfile as sf
from .offsets import RawOffsets
from .file_info import FileReaderInfo, StreamReaderInfo
from .utils import byte_array


__all__ = ['HeaderNotFoundError',
           'FileBase', 'VLBIFileReaderBase',
           'StreamBase', 'StreamReaderBase',
           'VLBIStreamReaderBase', 'StreamWriterBase',
           'FileInfo', 'FileOpener']


class HeaderNotFoundError(LookupError):
    """Error in finding a header in a stream."""
    pass


class FileBase:
    """File wrapper, used to add frame methods to a binary data file.

    The underlying file is stored in ``fh_raw`` and all attributes that do not
    exist on the class itself are looked up on it.

    Parameters
    ----------
    fh_raw : filehandle
        Filehandle of the raw binary data file.

    Notes
    -----
    A subclass for reading should define ``read_header``, ``read_frame``, and
    ``get_frame_rate`` methods, and also set an ``info`` property. Often the
    standard one will suffice, i.e., ``info = FileReaderInfo()``.

    A subclass for writing should define a ``write_frame`` method.

    """
    fh_raw = None

    def __init__(self, fh_raw):
        self.fh_raw = fh_raw

    def __getattr__(self, attr):
        """Try to get things on the current open file if it is not on self."""
        if not attr.startswith('_'):
            try:
                return getattr(self.fh_raw, attr)
            except AttributeError:
                pass
        #  __getattribute__ to raise appropriate error.
        return self.__getattribute__(attr)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fh_raw.close()

    @contextmanager
    def temporary_offset(self, offset=None, whence=0):
        """Context manager for temporarily seeking to another file position.

        To be used as part of a ``with`` statement::

            with fh_raw.temporary_offset() [as fh_raw]:
                with-block

        On exiting the ``with-block``, the file pointer is moved back to its
        original position.  As a convenience, one can pass on the offset
        to seek to when entering the context manager.  Parameters are as
        for :meth:`io.IOBase.seek`.
        """
        oldpos = self.tell()
        try:
            if offset is not None:
                self.seek(offset, whence)
            yield self
        finally:
            self.seek(oldpos)

    def __repr__(self):
        return "{0}(fh_raw={1})".format(self.__class__.__name__, self.fh_raw)

    def __getstate__(self):
        if self.writable():
            raise TypeError('cannot pickle file opened for writing')

        state = self.__dict__.copy()
        # IOBase instances cannot be pickled, but we can just reopen them
        # when we are unpickled.  Anything else may have internal state that
        # needs preserving (e.g., SequentialFile), so we will assume
        # it takes care of this itself.
        if isinstance(self.fh_raw, io.IOBase):
            fh = state.pop('fh_raw')
            state['fh_info'] = {
                'offset': 'closed' if fh.closed else fh.tell(),
                'filename': fh.name,
                'mode': fh.mode}

        return state

    def __setstate__(self, state):
        fh_info = state.pop('fh_info', None)
        if fh_info is not None:
            fh = io.open(fh_info['filename'], fh_info['mode'])
            if fh_info['offset'] != 'closed':
                fh.seek(fh_info['offset'])
            else:
                fh.close()
            state['fh_raw'] = fh

        self.__dict__.update(state)


class VLBIFileReaderBase(FileBase):
    """VLBI wrapped file reader base class.

    Parameters
    ----------
    fh_raw : filehandle
        Filehandle of the raw binary data file.

    Notes
    -----
    A subclass should define ``read_header`` and ``read_frame`` methods.

    This baseclass includes a base ``locate_frames`` method that can search
    the file for a header patter. It can be overridden by a version that just
    passes in the relevant pattern.

    Also defined is are a ``find_header`` method which combines the above with
    ``read_header``, and a basic ``get_frame_rate`` methods which scans the
    file for headers and determines the maximum frame number that occurs
    before the jump down (which is taken to be for the next second, as in all
    standard VLBI formats). This method assumes headers have a 'frame_nr'
    item, and define a ``payload_nbytes`` property.

    """

    info = FileReaderInfo()

    def locate_frames(self, pattern, *, mask=None, frame_nbytes=None,
                      offset=0, forward=True, maximum=None, check=1):
        """Use a pattern to locate frame starts near the current position.

        Note that the current position is always included.

        Parameters
        ----------
        pattern : header, ~numpy.ndaray, bytes, int, or iterable of int
            Synchronization pattern to look for.  If a header or header class,
            :meth:`~baseband.base.header.VLBIHeaderBase.invariant_pattern`
            is used to create a masked pattern, using invariant keys from
            :meth:`~baseband.base.header.VLBIHeaderBase.invariants`.
            If an `~numpy.ndarray` or `bytes` instance, a byte array view is
            taken. If an (iterable of) int, the integers need to be unsigned
            32 bit and will be interpreted as little-endian.
        mask : ~numpy.ndarray, bytes, int, or iterable of int.
            Bit mask for the pattern, with 1 indicating a given bit will
            be used the comparison.
        frame_nbytes : int, optional
            Frame size in bytes.  Defaults to the frame size in any header
            passed in.
        offset : int, optional
            Offset from the frame start that the pattern occurs.  Any
            offsets inferred from masked entries are added to this (hence,
            no offset needed when a header is passed in as ``pattern``).
        forward : bool, optional
            Seek forward if `True` (default), backward if `False`.
        maximum : int, optional
            Maximum number of bytes to search away from the present location.
            Default: search twice the frame size if given, otherwise 1 million
            (extra bytes to avoid partial patterns will be added).
            Use 0 to check only at the current position.
        check : int or tuple of int, optional
            Frame offsets where another sync pattern should be present
            (if inside the file). Ignored if ``frame_nbytes`` is not given.
            Default: 1, i.e., a sync pattern should be present one
            frame after the one found (independent of ``forward``),
            thus helping to guarantee the frame is not corrupted.

        Returns
        -------
        locations : list of int
            Locations of sync patterns within the range scanned,
            in order of proximity to the starting position.
        """
        if hasattr(pattern, 'invariant_pattern'):
            if frame_nbytes is None:
                frame_nbytes = pattern.frame_nbytes
            pattern, mask = pattern.invariant_pattern()

        pattern = byte_array(pattern)

        if mask is not None:
            mask = byte_array(mask)
            useful = np.nonzero(mask)[0]
            useful_slice = slice(useful[0], useful[-1]+1)
            mask = mask[useful_slice]
            pattern = pattern[useful_slice]
            offset += useful_slice.start

        if maximum is None:
            maximum = (2 * frame_nbytes if frame_nbytes else 1000000) - 1

        if check is None or frame_nbytes is None:
            check = np.array([], dtype=int)
            check_min = check_max = 0
        else:
            check = np.atleast_1d(check) * frame_nbytes
            # For Numpy >=1.15, can just be check.min(initial=0) in the
            # calculation of start, stop below.
            check_min = min(check.min(), 0)
            check_max = max(check.max(), 0)

        if frame_nbytes is None:
            # If not set, we just let it stand in for the extra bytes
            # that should be read to ensure we can even see a pattern.
            frame_nbytes = offset + pattern.size

        with self.temporary_offset() as fh:
            # Calculate the fiducial start of the region we are looking in.
            if forward:
                seek_start = fh.tell()
            else:
                seek_start = fh.tell() - maximum
            # Determine what part of the file to read, including the extra
            # bits for doing the checking.  Note that we ensure not to start
            # before the start of the file, but we do not check for ending
            # after the end of the file, as we want to avoid a possibly
            # expensive seek (e.g., if the file is a SequentialFile of a lot
            # of individual files).  We rely instead on reading fewer bytes
            # than requested if we are close to the end.
            start = max(seek_start + offset + check_min, 0)
            stop = max(seek_start + maximum + 1 + check_max + frame_nbytes,
                       start)

            fh.seek(start)
            data = fh.read(stop-start)

        # Since we may have hit the end of the file, check what the actual
        # stop position was (needed at end).
        stop = start + len(data)
        # We normally have read more than really needed (to check for EOF);
        # select what we actually want and convert to an array of bytes.
        size = min(maximum + 1 + check_max - check_min,
                   stop - start - pattern.size)
        if size <= 0:
            return []
        data = np.frombuffer(data, dtype='u1', count=size+pattern.size)

        # We match in two steps, first matching just the first pattern byte,
        # and then matching the rest in one step using a strided array.
        # The hope is that this gives a good compromise in speed: not
        # iterate in python for each pattern byte, yet not doing pattern
        # size times more comparisons than needed in C.
        if mask is None:
            match = data[:-pattern.size] == pattern[0]
        else:
            match = ((data[:-pattern.size] ^ pattern[0]) & mask[0]) == 0
        matches = np.nonzero(match)[0]
        # Re-stride data so that it looks like each element is followed by
        # all others, and check those all in one go.
        strided = as_strided(data[1:], strides=(1, 1),
                             shape=(size, pattern.size-1))
        if mask is None:
            match = strided[matches] == pattern[1:]
        else:
            match = ((strided[matches] ^ pattern[1:]) & mask[1:]) == 0
        matches = matches[match.all(-1)]

        if not forward:
            # Order by proximity to the file position.
            matches = matches[::-1]

        # Convert matches to a list of actual file positions.
        matches += start - offset
        matches = matches.tolist()
        # Keep only matches for which
        # (1) the location is in the requested range of current +/- maximum,
        # (2) the associated frames completely fit in the file, and
        # (3) the pattern is also present at the requested check points.
        # The range in positions actually wanted. Here, we guarantee not
        # only (1) but also (2) by using the recalculated `stop` from above,
        # which is limited by the file size.
        loc_start = max(seek_start, 0)
        loc_stop = min(seek_start+maximum+1, stop-frame_nbytes+1)
        # And the range in which it was possible to check positions.
        check_start = start
        check_stop = stop-offset-pattern.size
        locations = [loc for loc in matches
                     if (loc_start <= loc < loc_stop
                         and all(c in matches for c in loc+check
                                 if check_start <= c < check_stop))]

        return locations

    def find_header(self, *args, **kwargs):
        """Find the nearest header from the current position.

        If successful, the file pointer is left at the start of the header.

        Parameters are as for ``locate_frames``.

        Returns
        -------
        header
            Retrieved header.

        Raises
        ------
        ~baseband.base.base.HeaderNotFoundError
            If no header could be located.
        AssertionError
            If the header did not pass verification.
        """
        locations = self.locate_frames(*args, **kwargs)
        if not locations:
            raise HeaderNotFoundError('could not locate a a nearby frame.')
        self.seek(locations[0])
        with self.temporary_offset():
            return self.read_header()

    def get_frame_rate(self):
        """Determine the number of frames per second.

        The method cycles through headers, starting from the start of the file,
        finding the largest frame number before it jumps back to 0 for a new
        second.

        Returns
        -------
        frame_rate : `~astropy.units.Quantity`
            Frames per second.

        Raises
        ------
        `EOFError`
            If the file contains less than one second of data.
        """
        with self.temporary_offset(0):
            header = header0 = self.read_header()
            frame_nr0 = header0['frame_nr']
            while header['frame_nr'] == frame_nr0:
                self.seek(header.payload_nbytes, 1)
                header = self.read_header()
                max_frame = frame_nr0
            while header['frame_nr'] > 0:
                max_frame = max(header['frame_nr'], max_frame)
                self.seek(header.payload_nbytes, 1)
                header = self.read_header()

        return (max_frame + 1) * u.Hz


class StreamBase:
    """Baseband file wrapper, allowing access as a stream of data.

    Common methods between stream readers and writers, mostly just
    dealing with getting the arguments on initialization and providing
    access to them via properties.
    """
    # Apart from verify, instances are meant to be immutable.

    _sample_shape_maker = None
    _frame_index = None

    def __init__(self, fh_raw, header0, *, squeeze=True, **kwargs):
        # Required arguments.
        self.fh_raw = fh_raw
        self._header0 = header0
        # Arguments with defaults.
        self._squeeze = bool(squeeze)
        # Arguments that can override or complement information from header.
        for attr, getter in [
                ('bps', operator.index),
                ('complex_data', bool),
                ('samples_per_frame', operator.index),
                ('sample_shape', tuple),
                ('sample_rate', None)]:
            value = kwargs.pop(attr, None)
            if value is None:
                value = getattr(header0, attr, None)
            if getter is not None and value is not None:
                value = getter(value)
            setattr(self, '_'+attr, value)

        if kwargs:
            raise TypeError('got unexpected keyword(s): {}'
                            .format(', '.join(kwargs.keys())))
        # Pre-calculate.
        self._frame_rate = (self.sample_rate / self.samples_per_frame).to(u.Hz)
        # Initialize.
        self.offset = 0
        # Ensure that we have a sample_shape.
        self.sample_shape

    @property
    def squeeze(self):
        """Whether data arrays have dimensions with length unity removed.

        If `True`, data read out has such dimensions removed, and data
        passed in for writing has them inserted.
        """
        return self._squeeze

    @property
    def _unsliced_shape(self):
        unsliced_shape = self._sample_shape
        if self._sample_shape_maker is not None:
            return self._sample_shape_maker(*unsliced_shape)
        else:
            return unsliced_shape

    @lazyproperty
    def sample_shape(self):
        """Shape of a complete sample (possibly subset or squeezed)."""
        if not self.squeeze:
            return self._unsliced_shape

        sample_shape = self._unsliced_shape
        squeezed_shape = tuple(dim for dim in sample_shape if dim > 1)
        fields = getattr(sample_shape, '_fields', None)
        if fields is None:
            return squeezed_shape

        fields = [field for field, dim in zip(fields, sample_shape) if dim > 1]
        shape_cls = namedtuple('SampleShape', ','.join(fields))
        return shape_cls(*squeezed_shape)

    def _get_time(self, header):
        """Get time from a header."""
        # Subclasses can override this if information is needed beyond that
        # provided in the header.
        return header.time

    def _set_time(self, header, time):
        """Set time in a header."""
        # Subclasses can override this if information is needed beyond that
        # provided in the header.  Use update since that some classes will
        # do extra work after any setting (e.g., CRC update).
        header.update(time=time)

    def _get_index(self, header):
        """Infer the index of the frame header relative to the first frame."""
        # This base implementation uses the time, but can be overridden
        # with faster methods in subclasses.
        dt = self._get_time(header) - self.start_time
        return int(round((dt * self._frame_rate).to_value(u.one)))

    def _set_index(self, header, index):
        """Set frame header index relative to the first frame."""
        # Can be overridden if there is a simpler way than using the time,
        # or if other properties need to be set.
        self._set_time(header,
                       time=self.start_time + index / self._frame_rate)

    @lazyproperty
    def start_time(self):
        """Start time of the file.

        See also `time` for the time of the sample pointer's current offset.
        """
        return self._get_time(self.header0)

    @property
    def time(self):
        """Time of the sample pointer's current offset in file.

        See also `start_time` for the start time of the file.
        """
        return self.tell(unit='time')

    @property
    def header0(self):
        """First header of the file."""
        return self._header0

    @property
    def bps(self):
        """Bits per elementary sample."""
        return self._bps

    @property
    def complex_data(self):
        """Whether the data are complex."""
        return self._complex_data

    @property
    def samples_per_frame(self):
        """Number of complete samples per frame."""
        return self._samples_per_frame

    @property
    def sample_rate(self):
        """Number of complete samples per second."""
        return self._sample_rate

    def tell(self, unit=None):
        """Current offset in the file.

        Parameters
        ----------
        unit : `~astropy.units.Unit` or str, optional
            Time unit the offset should be returned in.  By default, no unit
            is used, i.e., an integer enumerating samples is returned. For the
            special string 'time', the absolute time is calculated.

        Returns
        -------
        offset : int, `~astropy.units.Quantity`, or `~astropy.time.Time`
             Offset in current file (or time at current position).
        """
        if unit is None:
            return self.offset

        # "isinstance" avoids costly comparisons of an actual unit with 'time'.
        if not isinstance(unit, u.UnitBase) and unit == 'time':
            return self.start_time + self.tell(unit=u.s)

        return (self.offset / self.sample_rate).to(unit)

    def __getattr__(self, attr):
        """Try to get things on the current open file if it is not on self."""
        if attr in {'readable', 'writable', 'seekable', 'closed', 'name'}:
            return getattr(self.fh_raw, attr)
        #  __getattribute__ to raise appropriate error.
        return self.__getattribute__(attr)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fh_raw.close()

    def __repr__(self):
        return ("<{s.__class__.__name__} name={s.name} offset={s.offset}\n"
                "    sample_rate={s.sample_rate},"
                " samples_per_frame={s.samples_per_frame},\n"
                "    sample_shape={s.sample_shape}, bps={s.bps},\n"
                "    {sub}start_time={s.start_time.isot}>"
                .format(s=self, sub=('subset={0}, '.format(self.subset)
                                     if self.subset else '')))


class StreamReaderBase(StreamBase):
    """Base for all stream readers, providing standard reading methods.

    Parameters
    ----------
    fh_raw : filehandle
        Should be opened in binary mode for reading, and usually wrapped
        in a given format;s ``FileReader``.
    header0 : header instance
        First header of the file, with information like the start time, etc.
    squeeze : bool, optional
        If `True` (default), remove any dimensions of length unity from
        decoded data.
    subset : indexing object or tuple of objects, optional
        Specific components of the complete sample to decode (after possible
        squeezing).
    fill_value : float or complex, optional
        Value to use for invalid or missing data. Default: 0.
    verify : bool, optional
        Whether to do basic checks of frame integrity when reading.  The first
        frameset of the stream is always checked.  Default: `True`.
    **kwargs
        Information that supplements that of the header. In particular, any
        format should define ``bps``, ``complex_data``, ``samples_per_frame``,
        and ``sample_shape`` (pre-squeeze and subset). Furthermore,
        ``sample_rate`` should be provided if the raw file reader cannot
        determine it.

    Notes
    -----
    Accessing frames happens via a number of private methods, which can be
    overridden by subclasses to deal with their peculiarities.

    The process starts in `~baseband.base.base.StreamReaderBase.read` with
    ``_get_frame(offset)``, which determines the ``index`` of the frame a
    given offset falls in, and then either uses ``_read_frame(index)`` to
    read it or returns a possibly cached frame, together with the
    ``sample_offset`` at which ``offset`` falls within the frame. Then, data
    is retrieved as needed until the frame is exhausted, at which point the
    process repeats.

    The ``_read_frame(index)`` method in turn takes a number of steps.
    First, the binary file reader is moved to the correct position using
    ``_seek_frame(index)``, then for the base reader, the actual reading is
    done using ``_fh_raw_read_frame()`` and a check is made that the right
    frame was read.

    Various formats override various steps. For instance, DADA overrides
    ``_fh_raw_read_frame()`` to take care of last frames, which are often
    longer than the fixed lengths recorded in the header. For GUPPI,
    ``_get_frame`` is overridden to take into account overlap between
    frames, which should not be skipped for the last frame.  For GSB,
    ``_seek_frame`` and ``_fh_raw_read_frame`` are both overridden to take
    into account that header and data are read from different files.
    """

    info = StreamReaderInfo()

    def __init__(self, fh_raw, header0, *,
                 squeeze=True, subset=(), fill_value=0., verify=True,
                 **kwargs):
        self._subset = (() if subset is None
                        else subset if isinstance(subset, tuple)
                        else (subset,))
        self._fill_value = float(fill_value)
        self.verify = verify
        super().__init__(fh_raw, header0, squeeze=squeeze, **kwargs)

    @lazyproperty
    def sample_rate(self):
        sample_rate = super().sample_rate
        if sample_rate is None:
            try:
                sample_rate = (self.samples_per_frame
                               * self.fh_raw.get_frame_rate()).to(u.MHz)

            except Exception as exc:
                exc.args += ("the sample rate could not be auto-detected. "
                             "This can happen if the file is too short to "
                             "determine the sample rate, or because it is "
                             "corrupted.  Try passing in an explicit "
                             "`sample_rate`.",)
                raise

        return sample_rate

    @property
    def verify(self):
        """Whether to do consistency checks on frames being read."""
        return self._verify

    @verify.setter
    def verify(self, verify):
        self._verify = bool(verify) if verify != 'fix' else verify

    @property
    def subset(self):
        """Specific components of the complete sample to decode.

        The order of dimensions is the same as for `sample_shape`.  Set by
        the class initializer.
        """
        return self._subset

    def _squeeze_and_subset(self, data):
        """Possibly remove unit dimensions and subset the given data.

        The first dimension (sample number) is never removed.
        """
        if self.squeeze:
            data = data.reshape(data.shape[:1]
                                + tuple(sh for sh in data.shape[1:] if sh > 1))
        if self.subset:
            data = data[(slice(None),) + self.subset]

        return data

    @lazyproperty
    def sample_shape(self):
        """Shape of a complete sample (possibly subset or squeezed)."""
        # First apply base class, which squeezes if needed.
        sample_shape = super().sample_shape
        if not self.subset:
            return sample_shape

        # Now apply subset to a dummy sample that has the sample number as its
        # value (where 13 is to bring bad luck to over-complicated subsets).
        dummy_data = np.arange(13.)
        dummy_sample = np.moveaxis(
            (np.zeros(sample_shape)[..., np.newaxis] + dummy_data), -1, 0)
        try:
            dummy_subset = dummy_sample[(slice(None),) + self.subset]
            # Check whether subset was in range and whether sample numbers were
            # preserved (latter should be, but check anyway).
            assert 0 not in dummy_subset.shape
            assert np.all(np.moveaxis(dummy_subset, 0, -1) == dummy_data)
        except (IndexError, AssertionError) as exc:
            exc.args += ("subset {} cannot be used to properly index "
                         "{}samples with shape {}.".format(
                             self.subset, "squeezed " if self.squeeze else "",
                             sample_shape),)
            raise exc

        # We got the shape.  We only bother trying to associate names with the
        # dimensions when we know for sure what happened in the subsetting.
        subset_shape = dummy_subset.shape[1:]
        if (not hasattr(sample_shape, '_fields') or subset_shape == ()
                or len(self.subset) > len(sample_shape)):
            return subset_shape

        # We can only associate names when indexing each dimension separately
        # gives a consistent result with the complete subset.
        subset_axis = 0
        fields = []
        subset = self.subset + (slice(None),) * (len(sample_shape)
                                                 - len(self.subset))
        try:
            for field, sample_dim, item in zip(sample_shape._fields,
                                               sample_shape, subset):
                subset_dim = np.empty(sample_dim)[item].shape
                assert len(subset_dim) <= 1  # No advanced multi-d indexing.
                if len(subset_dim) == 1:
                    # If this dimension was not removed and matches that
                    # of the real subset, we now have a field label for it.
                    assert subset_dim[0] == subset_shape[subset_axis]
                    fields.append(field)
                    subset_axis += 1
        except Exception:
            # Things did not make sense, probably some advanced indexing;
            # Just don't worry about having a named tuple.
            return subset_shape

        shape_cls = namedtuple('SampleShape', ','.join(fields))
        return shape_cls(*subset_shape)

    @lazyproperty
    def _last_header(self):
        """Header of the last file for this stream."""
        # Base implementation.  Seek forward rather than backward, as last
        # frame can have missing bytes.
        with self.fh_raw.temporary_offset() as fh_raw:
            file_size = fh_raw.seek(0, 2)
            nframes, fframe = divmod(file_size, self.header0.frame_nbytes)
            fh_raw.seek((nframes - 1) * self.header0.frame_nbytes)
            return fh_raw.read_header()

    # Override the following so we can refer to stop_time in the docstring.
    @property
    def start_time(self):
        """Start time of the file.

        See also `time` for the time of the sample pointer's current offset,
        and (if available) `stop_time` for the time at the end of the file.
        """
        return super().start_time

    @property
    def time(self):
        """Time of the sample pointer's current offset in file.

        See also `start_time` for the start time, and (if available)
        `stop_time` for the end time, of the file.
        """
        return super().time

    @lazyproperty
    def stop_time(self):
        """Time at the end of the file, just after the last sample.

        See also `start_time` for the start time of the file, and `time` for
        the time of the sample pointer's current offset.
        """
        return (self._get_time(self._last_header)
                + (self.samples_per_frame / self.sample_rate))

    @lazyproperty
    def _nsample(self):
        """Number of complete samples in the stream data."""
        return int(((self.stop_time - self.start_time)
                    * self.sample_rate).to(u.one).round())

    @property
    def shape(self):
        """Shape of the (squeezed/subset) stream data."""
        return (self._nsample,) + self.sample_shape

    @property
    def size(self):
        """Total number of component samples in the (squeezed/subset) stream
        data.
        """
        prod = 1
        for dim in self.shape:
            prod *= dim
        return prod

    @property
    def ndim(self):
        """Number of dimensions of the (squeezed/subset) stream data."""
        return len(self.shape)

    @property
    def fill_value(self):
        """Value to use for invalid or missing data. Default: 0."""
        return self._fill_value

    @property
    def dtype(self):
        # TODO: arguably, this should be inferred from an actual payload.
        return np.dtype(np.complex64 if self.complex_data else np.float32)

    def readable(self):
        """Whether the file can be read and decoded."""
        return self.fh_raw.readable and self.fh_raw.info.readable

    def seek(self, offset, whence=0):
        """Change the sample pointer position.

        This works like a normal filehandle seek, but the offset is in samples
        (or a relative or absolute time).

        Parameters
        ----------
        offset : int, `~astropy.units.Quantity`, or `~astropy.time.Time`
            Offset to move to.  Can be an (integer) number of samples,
            an offset in time units, or an absolute time.  For the latter
            two, the pointer will be moved to the nearest integer sample.
        whence : {0, 1, 2, 'start', 'current', or 'end'}, optional
            Like regular seek, the offset is taken to be from the start if
            ``whence=0`` (default), from the current position if 1,
            and from the end if 2.  One can alternativey use 'start',
            'current', or 'end' for 0, 1, or 2, respectively.  Ignored if
            ``offset`` is a time.
        """
        try:
            offset = operator.index(offset)
        except Exception:
            try:
                offset = offset - self.start_time
            except Exception:
                pass
            else:
                whence = 0

            offset = int((offset * self.sample_rate).to(u.one).round())

        if whence == 0 or whence == 'start':
            self.offset = offset
        elif whence == 1 or whence == 'current':
            self.offset += offset
        elif whence == 2 or whence == 'end':
            self.offset = self.shape[0] + offset
        else:
            raise ValueError("invalid 'whence'; should be 0 or 'start', 1 or "
                             "'current', or 2 or 'end'.")

        return self.offset

    def read(self, count=None, out=None):
        """Read a number of complete samples.

        Parameters
        ----------
        count : int or None, optional
            Number of complete samples to read. If `None` (default) or
            negative, the number of samples left. Ignored if ``out`` is given.
        out : None or array, optional
            Array to store the samples in. If given, ``count`` will be inferred
            from the first dimension; the remaining dimensions should equal
            `sample_shape`.

        Returns
        -------
        out : `~numpy.ndarray` of float or complex
            The first dimension is sample-time, and the remaining ones are
            as given by `sample_shape`.
        """
        if self.closed:
            raise ValueError("I/O operation on closed stream.")

        samples_left = self.shape[0] - self.offset
        if out is None:
            if count is None or count < 0:
                count = max(0, samples_left)

            out = np.empty((count,) + self.sample_shape, dtype=self.dtype)
        else:
            assert out.shape[1:] == self.sample_shape, (
                "'out' must have trailing shape {}".format(self.sample_shape))
            count = out.shape[0]

        if count > samples_left:
            raise EOFError("cannot read from beyond end of input.")

        offset0 = self.offset
        sample = 0
        while sample < count:
            # For current position, get frame plus offset in that frame.
            frame, sample_offset = self._get_frame(self.offset)
            nsample = min(count - sample, len(frame) - sample_offset)
            data = frame[sample_offset:sample_offset + nsample]
            data = self._squeeze_and_subset(data)
            # Copy to relevant part of output.
            out[sample:sample + nsample] = data
            sample += nsample
            # Explicitly set offset (leaving get_frame free to adjust it).
            self.offset = offset0 + sample

        return out

    def _get_frame(self, offset):
        """Get a frame that includes given offset.

        Finds the index corresponding to the needed frame, assuming frames
        are all the same length.  If not already cached, it retrieves a
        frame by calling ``self._read_frame(index)``.

        Parameters
        ----------
        offset : int
            Offset in the stream for which a frame should be found.

        Returns
        -------
        frame : `~baseband.base.frame.FrameBase`
            Frame holding the sample at ``offset``.
        sample_offset : int
            Offset within the frame corresponding to ``offset``.
        """
        frame_index, sample_offset = divmod(offset, self.samples_per_frame)
        if frame_index != self._frame_index:
            self._frame = self._read_frame(frame_index)
            self._frame.fill_value = self.fill_value
            self._frame_index = frame_index

        return self._frame, sample_offset

    def _read_frame(self, index):
        """Base implementation of reading a frame.

        It cannot handle bad frames, and contains pieces which subclasses
        can override as needed (or override the whole thing).
        """
        self._seek_frame(index)
        frame = self._fh_raw_read_frame()
        # Check whether we actually got the right frame.
        if self.verify and self._get_index(frame) != index:
            raise ValueError('wrong frame number.')

        return frame

    def _seek_frame(self, index):
        """Move the underlying file pointer to the frame of the given index."""
        return self.fh_raw.seek(index * self.header0.frame_nbytes)

    def _fh_raw_read_frame(self):
        """Read a frame at the current position of the underlying file."""
        return self.fh_raw.read_frame(verify=self.verify)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove cached frame and associate index, since these
        # are almost certainly not needed and can be regenerated.
        # Sample_shape can also be regenerated, yet not easily pickled,
        # if the namedtuple is defined on the class, so we remove it too.
        for item in ('_frame', '_frame_index', 'sample_shape'):
            state.pop(item, None)

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


class VLBIStreamReaderBase(StreamReaderBase):
    """Base for VLBI stream readers.

    This extends the base class specifically for the case that the underlying
    file has small subsequent frames, which may have gaps, and adds methods
    that allow fixing these.

    Arguments are as for `~baseband.base.base.StreamReaderBase`.

    Notes
    -----
    The code assumes that the raw file reader has a ``find_header`` method,
    which can be used to find next or previous frames from any position, and
    thus help identify and skip bad frames. It is used in ``_last_header``
    to be sure to find a full last frame.

    The class also overrides the ``_read_frame`` method which one that checks
    that the frame after the one requested is the right one, and attemps
    recovery if anything fails, using ``_bad_frame(index, frame, exc)``.

    Subclasses can overrides parts of the class. For instance, VDIF overrides
    ``_fh_raw_read_frame`` and ``_bad_frame`` to take care of the fact that
    data comes in sets of frames from multiple threads, not individual ones.
    """

    _next_index = None

    def __init__(self, fh_raw, header0, **kwargs):
        super().__init__(fh_raw, header0, **kwargs)
        self._raw_offsets = RawOffsets(frame_nbytes=self.header0.frame_nbytes)

    @lazyproperty
    def _last_header(self):
        """Last header of the file."""
        with self.fh_raw.temporary_offset(
                -self.header0.frame_nbytes, 2) as fh_raw:
            try:
                return fh_raw.find_header(self.header0, forward=False,
                                          check=(-1, 1))
            except HeaderNotFoundError as exc:
                exc.args += ("corrupt VLBI frame? No frame in last {0} bytes."
                             .format(2 * self.header0.frame_nbytes),)
                raise

    def _seek_frame(self, index):
        """Move the underlying file pointer to the frame of the given index."""
        return self.fh_raw.seek(self._raw_offsets[index])

    def _read_frame(self, index):
        """Base implementation of reading a VLBI frame.

        It can handle bad frames to some extent, and contains pieces
        which subclasses can override as needed.
        """
        self._seek_frame(index)
        if not self.verify:
            return self._fh_raw_read_frame()

        # If we are reading with care, we read also a frame ahead
        # to make sure that is not corrupted.  If such a frame has
        # been kept, we use it now.  Otherwise, we read a new frame.
        # We always remove the cached copy, since we cannot be sure
        # it is even the right number.
        if index == self._next_index:
            frame = self._next_frame
            frame_index = index
            self.fh_raw.seek(frame.nbytes, 1)
            self._next_index = self._next_frame = None

        else:
            self._next_index = self._next_frame = None
            try:
                frame = self._fh_raw_read_frame()
                frame_index = self._get_index(frame)
            except Exception as exc:
                return self._bad_frame(index, None, exc)

        # Check whether we actually got the right frame.
        if frame_index != index:
            return self._bad_frame(index, frame,
                                   ValueError('wrong frame number.'))

        # In either case, we check the next frame (if there is one).
        try:
            with self.fh_raw.temporary_offset():
                self._next_frame = self._fh_raw_read_frame()
                self._next_index = self._get_index(self._next_frame)
        except Exception as exc:
            return self._bad_frame(index, frame, exc)

        return frame

    def _bad_frame(self, index, frame, exc):
        """Deal with a bad frame.

        Parameters
        ----------
        index : int
            Frame index that is to be read.
        frame : `~baseband.base.frame.VLBIFrameBase` or None
            Frame that was read without failure.  If not `None`, either
            the frame index is wrong or the next frame could not be read.
        exc : Exception
            Exception that led to the call.
        """
        if (frame is not None and self._get_index(frame) == index
                and index == self._get_index(self._last_header)):
            # If we got an exception because we're trying to read beyond the
            # last frame, the frame is almost certainly OK, so keep it.
            return frame

        if self.verify != 'fix':
            raise exc

        msg = 'problem loading frame {}.'.format(index)

        # Where should we be?
        raw_offset = self._seek_frame(index)
        # See if we're in the right place.  First ensure we have a header.
        # Here, it is more important that it is a good one than that we go
        # too far, so we insist on two consistent frames after it.  We
        # increase the maximum a bit to be able to jump over bad bits.
        self.fh_raw.seek(raw_offset)
        try:
            header = self.fh_raw.find_header(
                self.header0, forward=True, check=(1, 2),
                maximum=3*self.header0.frame_nbytes)
        except HeaderNotFoundError:
            exc.args += (msg + ' Cannot find header nearby.',)
            raise exc

        # Don't yet know how to deal with excess data.
        header_index = self._get_index(header)
        if header_index < index:
            exc.args += (msg + ' There appears to be excess data.',)
            raise exc

        # Go backward until we find previous frame, storing offsets
        # as we go.  We again increase the maximum since we may need
        # to jump over a bad bit.
        while header_index >= index:
            raw_pos = self.fh_raw.tell()
            header1_index = header_index
            self.fh_raw.seek(-1, 1)
            try:
                header = self.fh_raw.find_header(
                    self.header0, forward=False,
                    maximum=4*self.header0.frame_nbytes)
            except HeaderNotFoundError:
                exc.args += (msg + ' Could not find previous index.',)
                raise exc

            header_index = self._get_index(header)
            # While we are at it, update the list of known indices.
            self._raw_offsets[header1_index] = raw_pos

        # Move back to position of last good header (header1_index).
        self.fh_raw.seek(raw_pos)

        if header1_index > index:
            # Frame is missing!
            msg += ' The frame seems to be missing.'
            # Just reuse old frame with new index and set to invalid.
            frame = self._frame
            frame.header.mutable = True
            frame.valid = False
            self._set_index(frame, index)

        else:
            assert header1_index == index, \
                'at this point, we should have a good header.'
            if raw_pos != raw_offset:
                msg += ' Stream off by {0} bytes.'.format(raw_offset
                                                          - raw_pos)
                # Above, we should have added information about
                # this index in our offset table.
                assert index in self._raw_offsets.frame_nr

            # At this point, reading the frame should always work,
            # and we know there is a header right after it.
            frame = self._fh_raw_read_frame()
            assert self._get_index(frame) == index

        warnings.warn(msg)
        return frame

    def __getstate__(self):
        state = super().__getstate__()
        # Remove additional cached frames and indices.
        for item in ('_next_frame', '_next_index'):
            state.pop(item, None)

        return state


class StreamWriterBase(StreamBase):
    """Base for all stream writers, providing a standard write method.

    Parameters
    ----------
    fh_raw : filehandle
        Should be opened in binary mode for writer, and usually wrapped
        in a given format's ``FileWriter``.
    header0 : header instance
        Header for the first frame, holding time information, etc.  Can instead
        give keyword arguments to construct a header (see ``**kwargs``).
    squeeze : bool, optional
        If `True` (default), writer accepts squeezed arrays as input, and adds
        any dimensions of length unity.
    **kwargs
        Information that supplements that of the header. In particular, any
        format should define ``bps``, ``complex_data``, ``samples_per_frame``,
        ``sample_shape`` (on file), and ``sample_rate``.

    Notes
    -----
    Accessing frames happens via a number of private methods, which can be
    overridden by subclasses to deal with their  peculiarities.

    The process starts in `~baseband.base.base.StreamWriterBase.write` with
    ``_get_frame(offset)``, which determines the ``index`` of the frame a
    given offset falls in, and then either uses ``_make_frame(index)`` to
    create it, or returns a possibly cached frame, together with the
    ``sample_offset`` at which ``offset`` falls within the frame. Then, data
    is inserted as needed until the frame is exhausted, at which point the
    frame is written to disk using ``_fh_raw_write_frame`` and the process
    repeats.

    Various formats override various steps. For instance, for DADA and GUPPI,
    ``_make_frame()`` and ``_fh_raw_write_frame()`` are overridden to work
    with memory mapped files. For GSB, ``_fh_raw_write_frame()`` is overridden
    to take into account that header and data are read from different files.
    """

    def _unsqueeze(self, data):
        new_shape = list(data.shape)
        for i, dim in enumerate(self._unsliced_shape):
            if dim == 1:
                new_shape.insert(i + 1, 1)
        return data.reshape(new_shape)

    def write(self, data, valid=True):
        """Write data, buffering by frames as needed.

        Parameters
        ----------
        data : `~numpy.ndarray`
            Piece of data to be written, with sample dimensions as given by
            `sample_shape`. This should be properly scaled to make best use
            of the dynamic range delivered by the encoding.
        valid : bool, optional
            Whether the current data are valid.  Default: `True`.
        """
        assert data.shape[1:] == self.sample_shape, (
            "'data' should have trailing shape {}".format(self.sample_shape))

        if self.squeeze:
            data = self._unsqueeze(data)

        count = data.shape[0]
        offset0 = self.offset
        sample = 0
        while sample < count:
            frame, sample_offset = self._get_frame(self.offset)
            nsample = min(count - sample, len(frame) - sample_offset)
            sample_end = sample_offset + nsample
            frame[sample_offset:sample_end] = data[sample:sample+nsample]
            frame.valid &= valid
            if sample_end == len(frame):
                self._fh_raw_write_frame(frame)

            sample += nsample
            # Explicitly set offset (leaving get_frame free to adjust it).
            self.offset = offset0 + sample

    def _get_frame(self, offset):
        # Nearly identical to StreamReaderBase version, but not
        # quite worth separating it out.
        frame_index, sample_offset = divmod(offset, self.samples_per_frame)
        if frame_index != self._frame_index:
            self._frame = self._make_frame(frame_index)
            self._frame_index = frame_index

        return self._frame, sample_offset

    _get_frame.__doc__ = StreamReaderBase._get_frame.__doc__

    def _make_frame(self, index):
        # Default implementation assumes that an initial _frame was
        # set up and just re-uses it with a new index.
        self._set_index(self._frame, index)
        self._frame.valid = True
        return self._frame

    def _fh_raw_write_frame(self, frame):
        # Default implementation is to assume that the frame knows how to
        # write itself to the underlying file.
        frame.tofile(self.fh_raw)

    def close(self):
        extra = self.offset % self.samples_per_frame
        if extra != 0:
            warnings.warn("closing with partial buffer remaining.  "
                          "Writing padded frame, marked as invalid.")
            self.write(np.zeros((self.samples_per_frame - extra,)
                                + self.sample_shape), valid=False)
            assert self.offset % self.samples_per_frame == 0
        return super().close()


class FileInfo:
    """File information collector.

    The instance can be used as a function on a file name to get
    information from that file, by opening it and retrieving ``info``.

    Parameters
    ----------
    opener : callable
        The function to use to open files

    Notes
    -----
    The class is perhaps most easily used via the class method
    `~baseband.base.base.FileInfo.create`.
    """

    def __init__(self, opener):
        self.open = opener

    def _get_info(self, name, mode, **kwargs):
        """Open a file in the given mode and retrieve info."""
        try:
            with self.open(name, mode=mode, **kwargs) as fh:
                return fh.info
        except Exception as exc:
            return exc

    def is_ok(self, info):
        """Wether the item returned by _get_info has valid information."""
        return not isinstance(info, Exception) and info

    def get_file_info(self, name, **kwargs):
        """Open a file in binary mode and retrieve info.

        Any keyword arguments that were required to open the file will
        be stored as a ``used_kwargs`` attribute on the returned ``info``.

        Parameters
        ----------
        name : str or filehandle
            Item to be opened for reading in binary mode.
        **kwargs
            Any keyword arguments that might be required to open the
            file successfully (e.g., ``decade`` for Mark 4).

        Returns
        -------
        info : `~baseband.base.file_info.FileReaderInfo`
            Information on the file.  Will evaluate as `False` if the
            file was not in the right format.

        Notes
        -----
        Getting information should never fail. If an `Exception` is
        raised or returned, it is a bug in the file reader.
        """
        info = self._get_info(name, 'rb')
        # If right format, check if arguments were missing.
        if self.is_ok(info):
            used_kwargs = {key: kwargs[key] for key in info.missing
                           if key in kwargs}
            if used_kwargs:
                info = self._get_info(name, mode='rb', **used_kwargs)

            info.used_kwargs = used_kwargs

        return info

    def get_stream_info(self, name, file_info, **kwargs):
        """Open a file in stream mode and retrieve info.

        Any keyword arguments that were required to open the file will
        be stored as a ``used_kwargs`` attribute on the returned ``info``.

        Parameters
        ----------
        name : str or filehandle
            Item to be opened for reading in stream mode.
        file_info : `~baseband.base.file_info.FileReaderInfo`
            Information gleaned from opening in binary mode.
        **kwargs
            Any keyword arguments that might be required to open the
            file successfully (e.g., ``decade`` for Mark 4).

        Returns
        -------
        info : `~baseband.base.file_info.StreamReaderInfo`
            Information on the file.  Will evaluate as `False` if the
            file was not in the right format. Will return `None` if no
            sample rate information was present, or an `Exception` if
            the opening as a stream failed.
        """
        frame_rate = file_info.frame_rate
        used_kwargs = file_info.used_kwargs
        if frame_rate is None:
            if 'sample_rate' in kwargs:
                used_kwargs['sample_rate'] = kwargs['sample_rate']
            else:
                # frame rate will already be marked as missing in
                # file_info.
                return None

        stream_info = self._get_info(name, mode='rs', **used_kwargs)
        if self.is_ok(stream_info):
            stream_info.used_kwargs = used_kwargs

        return stream_info

    def __call__(self, name, **kwargs):
        """Collect baseband file information.

        First try opening as a binary file and check whether the file is
        of the correct format. If so, and no required information is missing,
        re-open as a stream, and get information like the start time,
        sample rate, etc.

        Parameters
        ----------
        name : str or filehandle, or sequence of str
            File name, filehandle, or sequence of file names.
        **kwargs
            Any other arguments the opener needs to open as a stream.

        Returns
        -------
        info
            :class:`~baseband.base.file_info.FileReaderInfo` or
            :class:`~baseband.base.file_info.StreamReaderInfo`.
            In addition to the normal ``info`` attributes, also stored
            are attributes about what happened to the keyword arguments:
            ``used_kwargs``, ``consistent_kwargs``, ``inconsistent_kwargs``
            and ``irrelevant_kwargs``.
        """
        # NOTE: getting info should never fail or even emit warnings.
        # Hence, warnings or errors should not be suppressed here, but
        # rather in the info implementations.
        file_info = self.get_file_info(name, **kwargs)
        if not file_info or file_info.missing:
            return file_info

        stream_info = self.get_stream_info(name, file_info, **kwargs)
        if not self.is_ok(stream_info):
            if isinstance(stream_info, Exception):
                # Unexpected errors.  Put it in file_info so there is a record.
                file_info.errors['stream'] = str(stream_info)
            return file_info

        self.check_consistency(stream_info, **kwargs)
        return stream_info

    def check_consistency(self, info, **kwargs):
        """Check consistency between info and the given arguments.

        The keyword arguments will be sorted into those that were used
        by the file opener and those that were unused, with the latter
        split in those that had consistent, inconsistent, or irrelevant
        information.  They are stored on the ``info`` instance in
        ``used_kwargs``, ``consistent_kwargs``, ``inconsistent_kwargs``
        and ``irrelevant_kwargs`` attributes, respectively.

        Parameters
        ----------
        info : `~baseband.base.file_info.StreamReaderInfo`
            Information gleaned from a file opened in stream reading mode.
        **kwargs
            Keyword arguments passed to the opener.
        """
        # Store what happened to the kwargs, so one can decide if there are
        # inconsistencies or other problems.
        info.consistent_kwargs = {}
        info.inconsistent_kwargs = {}
        info.irrelevant_kwargs = {}
        for key, value in kwargs.items():
            if key in info.used_kwargs:
                continue

            consistent = self.check_key(key, value, info)

            if consistent is None:
                info.irrelevant_kwargs[key] = value
            elif consistent:
                info.consistent_kwargs[key] = value
            else:
                info.inconsistent_kwargs[key] = value

        return info

    def check_key(self, key, value, info):
        """Check consistency for a given key and value.

        Parameters
        ----------
        key : str
            Name of the key.
        value : object
            Corresponding value.
        info : `~baseband.base.file_info.StreamReaderInfo`
            Information collected by opening a file in stream reader mode.

        Returns
        -------
        consistent : True, False, or None
            Whether the information on ``info`` for ``key`` is consistent
            with ``value``.  `None` if it could not be determined.
        """
        info_value = getattr(info, key, None)
        if info_value is None:
            info_value = getattr(info.file_info, key, None)

        if info_value is not None:
            return info_value == value

        if key == 'nchan':
            sample_shape = info.shape[1:]
            if sample_shape is not None:
                # If we passed nchan, and info doesn't have it, but does have a
                # sample shape, check that consistency with that, either in
                # being equal to `sample_shape.nchan` or equal to the product
                # of all elements (e.g., a VDIF file with 8 threads and 1
                # channel per thread is consistent with nchan=8).
                return (getattr(sample_shape, 'nchan', -1) == value
                        or np.prod(sample_shape) == value)

        elif key in {'ref_time', 'kday', 'decade'}:
            start_time = info.start_time
            if start_time is not None:
                if key == 'ref_time':
                    return abs(value - start_time).jd < 500
                elif key == 'kday':
                    return int(start_time.mjd / 1000.) * 1000 == value
                else:  # decade
                    return int(start_time.isot[:3]) * 10 == value

        return None

    def wrapped(self, module=None, doc=None):
        """Wrap as a function named info, replacing docstring and module."""

        @functools.wraps(self.__call__)
        def info(*args, **kwargs):
            return self(*args, **kwargs)

        if doc:
            info.__doc__ = doc

        # This ensures the function becomes visible to sphinx.
        if module:
            info.__module__ = module

        return info

    @classmethod
    def create(cls, ns):
        """Create an info getter for the given namespace.

        This assumes that the namespace contains an ``open`` function, which
        is used to create an instance of the info class that is wrapped in a
        function with ``__module__`` set to the calling module (inferred
        from the namespace).

        Parameters
        ----------
        ns : dict
            Namespace to look in.  Generally, pass in ``globals()`` at the
            call site.
        """
        module = ns.get('__name__', None)
        for key in ns:
            if key.endswith('StreamReader'):
                fmt = key.replace('StreamReader', '')
                break
        else:  # pragma: no cover
            fmt = None

        opener = ns['open']
        info = cls(opener)
        doc = textwrap.dedent(info.__call__.__doc__)
        if (fmt is not None
                and info.__call__.__doc__ is FileInfo.__call__.__doc__):
            doc = doc.replace(
                'Collect baseband file information.',
                f'Collect {fmt} file information.')
        return info.wrapped(module=module, doc=doc)


class FileOpener:
    """File opener for a baseband format.

    Each instance can be used as a function to open a baseband stream.
    It is probably best used inside a wrapper, so that the documentation
    can reflect the docstring of ``__call__`` rather than of this class.

    Parameters
    ----------
    fmt : str
        Name of the baseband format
    classes : dict
        With the file/stream reader/writer classes keyed by names equal to
        'FileReader', 'FileWriter', 'StreamReader', 'StreamWriter' prefixed
        by ``fmt``.  Typically, one will pass in ``classes=globals()``.
    header_class : `~baseband.base.header.VLBIHeaderBase` subclass
        Used to instantiate a header from keywords as needed.
    """

    FileNameSequencer = sf.FileNameSequencer
    """Sequencer used for templates."""

    non_header_keys = {'squeeze', 'subset', 'fill_value', 'verify',
                       'file_size'}
    """keyword arguments that should never be used to create a header."""

    _name = None

    def __init__(self, fmt, classes, header_class):
        self.fmt = fmt
        self.classes = classes
        self.header_class = header_class

    def normalize_mode(self, mode):
        if mode in self.classes:
            return mode
        if mode[::-1] in self.classes:
            return mode[::-1]
        if mode in {'r', 'w'}:
            return mode + 's'

        raise ValueError(f'invalid mode: {mode} '
                         f'({self.fmt} supports {set(self.classes)}).')

    def _get_type(self, name):
        if hasattr(name, 'read') or hasattr(name, 'write'):
            return 'fh'

        try:
            f0 = name[0]
        except (TypeError, IndexError):
            raise ValueError("name '{name}' not understood.") from None

        if (isinstance(name, (tuple, list, sf.FileNameSequencer))
                or isinstance(f0, str) and len(f0) > 1):
            return 'sequence'

        if '{' in name and '}' in name:
            return 'template'
        else:
            return 'name'

    def get_type(self, name):
        """Infer the type of file name is pointing to.

        Options are 'fh', 'name', 'sequence', and 'template'.
        """
        if self._name is not name:
            self._type = self._get_type(name)
            self._name = name
        return self._type

    def is_sequence(self, name):
        """Whether name is an (implied) sequence of files."""
        return self.get_type(name) in ('template', 'sequence')

    def is_template(self, name):
        """Whether name is a template for a sequence of files."""
        return self.get_type(name) == 'template'

    def is_name(self, name):
        """Whether name is a name of a file."""
        return self.get_type(name) == 'name'

    def is_fh(self, name):
        """Whether name is a filehandle."""
        return self.get_type(name) == 'fh'

    def get_header0(self, kwargs):
        """Get header0 from kwargs or construct it from kwargs.

        Possible keyword arguments will be popped from kwargs.
        """
        header0 = kwargs.get('header0', None)
        if header0 is None:
            tried = {key: value for key, value in kwargs.items()
                     if key not in self.non_header_keys}
            with warnings.catch_warnings():
                # Ignore possible warnings about extraneous arguments.
                warnings.simplefilter('ignore')
                header0 = self.header_class.fromvalues(**tried)
            # Pop the kwargs that we likely used.  We do this by inspection,
            # but note that this may still let extraneous keywords
            # by eaten up by header classes that just store everything.
            maybe_used = (
                set(inspect.signature(self.header_class.fromvalues).parameters)
                | set(self.header_class._properties)
                | set(header0.keys()))

            maybe_used = {key.lower() for key in maybe_used}
            used = set(key for key in tried if key.lower() in maybe_used)
            for key in used:
                kwargs.pop(key)

        return header0

    def get_fns(self, name, mode, kwargs):
        """Convert a template into a file-name sequencer.

        Any keywords needed to fill the template are popped from kwargs.
        """
        try:
            fns_kwargs = dict(self.get_header0(kwargs))
        except Exception:
            fns_kwargs = {}

        fns_kwargs.update(kwargs)
        fns = self.FileNameSequencer(name, fns_kwargs)
        for key in set(fns.items).intersection(kwargs):
            kwargs.pop(key)
        return fns

    def get_fh(self, name, mode, kwargs={}):
        """Ensure name is a filehandle, opening it if necessary."""
        if mode == 'wb' and self.is_sequence(name):
            raise ValueError(f"{self.fmt} does not support writing to a "
                             f"sequence or template in binary mode.")

        if self.is_fh(name):
            return name

        if self.is_template(name):
            name = self.get_fns(name, mode, kwargs)

        open_kwargs = {'mode': (mode[0].replace('w', 'w+')
                                + mode[1].replace('s', 'b'))}
        if self.is_sequence(name):
            opener = sf.open
            if 'file_size' in kwargs:
                open_kwargs['file_size'] = kwargs.pop('file_size')

        else:
            opener = io.open

        return opener(name, **open_kwargs)

    def __call__(self, name, mode='rs', **kwargs):
        """
        Open baseband file(s) for reading or writing.

        Opened as a binary file, one gets a wrapped filehandle that adds
        methods to read/write a frame.  Opened as a stream, the handle is
        wrapped further, with methods such as reading and writing to the file
        as if it were a stream of samples.

        Parameters
        ----------
        name : str or filehandle, or sequence of str
            File name, filehandle, sequence of file names, or template.
        mode : {'rb', 'wb', 'rs', or 'ws'}, optional
            Whether to open for reading or writing, and as a regular binary
            file or as a stream. Default: 'rs', for reading a stream.
        **kwargs
            Additional arguments when opening the file as a stream.
        """
        mode = self.normalize_mode(mode)
        if mode == 'ws':
            # Stream writing always needs a header.  Construct from
            # other keywords if necessary.
            kwargs['header0'] = self.get_header0(kwargs)

        fh = self.get_fh(name, mode, kwargs)
        try:
            return self.classes[mode](fh, **kwargs)
        except Exception:
            if fh is not name:
                fh.close()
            raise

    def wrapped(self, module=None, doc=None):
        """Wrap as a function named open, replacing docstring and module."""

        @functools.wraps(self.__call__)
        def open(*args, **kwargs):
            return self(*args, **kwargs)

        if doc:
            open.__doc__ = doc

        # This ensures the function becomes visible to sphinx.
        if module:
            open.__module__ = module

        return open

    @classmethod
    def create(cls, ns, doc=None):
        """Create a standard opener for the given namespace.

        This assumes that the namespace contains file and stream readers
        and writers, as well as a header class, with standard names,
        ``<fmt>FileReader``, ``<fmt>FileWriter``, ``<fmt>StreamReader``,
        ``<fmt>StreamWriter``, and ``<fmt>Header``, where ``fmt`` is the
        name of the format (which is inferred by looking for a
        ``*StreamReader`` entry).

        The opener is instantiated using the format and the above classes,
        and then a wrapping function is created with ``__module__`` set to
        the ``__name__`` of the namespace, and with the documentation of
        its ``__call__`` method extended with ``doc``.

        Parameters
        ----------
        ns : dict
            Namespace to look in.  Generally, pass in ``globals()`` at the
            call site.
        doc : str, optional
            Extra documentation to add to that of the opener's ``__call__``
            method.
        """
        module = ns.get('__name__', None)
        for key in ns:
            if key.endswith('StreamReader'):
                fmt = key.replace('StreamReader', '')
                break
        else:  # noqa
            raise ValueError('namespace does not contain a StreamReader, '
                             'so fmt cannot be guessed.')

        classes = {mode: ns[fmt + cls_type] for (mode, cls_type) in {
            'rb': 'FileReader',
            'wb': 'FileWriter',
            'rs': 'StreamReader',
            'ws': 'StreamWriter'}.items()}
        header_class = ns.get(fmt+'Header')
        opener = cls(fmt, classes, header_class)
        if doc is not None:
            doc = textwrap.dedent(opener.__call__.__doc__) + doc
            if (opener.__call__.__doc__ is FileOpener.__call__.__doc__):
                doc = doc.replace(
                    'Open baseband file(s) for reading or writing.',
                    f'Open {fmt} file(s) for reading or writing.')
        return opener.wrapped(module=module, doc=doc)
