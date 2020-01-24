# Licensed under the GPLv3 - see LICENSE
import io
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
from .file_info import VLBIFileReaderInfo, VLBIStreamReaderInfo
from .utils import byte_array


__all__ = ['HeaderNotFoundError',
           'VLBIFileBase', 'VLBIFileReaderBase', 'VLBIStreamBase',
           'VLBIStreamReaderBase', 'VLBIStreamWriterBase',
           'make_opener']


class HeaderNotFoundError(LookupError):
    """Error in finding a header in a stream."""
    pass


class VLBIFileBase:
    """VLBI file wrapper, used to add frame methods to a binary data file.

    The underlying file is stored in ``fh_raw`` and all attributes that do not
    exist on the class itself are looked up on it.

    Parameters
    ----------
    fh_raw : filehandle
        Filehandle of the raw binary data file.
    """

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
    def temporary_offset(self):
        """Context manager for temporarily seeking to another file position.

        To be used as part of a ``with`` statement::

            with fh_raw.temporary_offset() [as fh_raw]:
                with-block

        On exiting the ``with-block``, the file pointer is moved back to its
        original position.
        """
        oldpos = self.tell()
        try:
            yield self
        finally:
            self.seek(oldpos)

    def __repr__(self):
        return "{0}(fh_raw={1})".format(self.__class__.__name__, self.fh_raw)


class VLBIFileReaderBase(VLBIFileBase):
    """VLBI wrapped file reader base class.

    Typically, a subclass will define ``read_header``, ``read_frame``,
    and ``find_header`` methods.  This baseclass includes a `get_frame_rate`
    method which determines the frame rate by scanning the file for headers,
    looking for the maximum frame number that occurs before the jump down
    for the next second. This method requires the subclass to define a
    ``read_header`` method and assumes headers have a 'frame_nr' item, and
    define a ``payload_nbytes`` property (as do all standard VLBI formats).

    Parameters
    ----------
    fh_raw : filehandle
        Filehandle of the raw binary data file.
    """

    info = VLBIFileReaderInfo()

    def locate_frames(self, pattern, *, mask=None, frame_nbytes=None,
                      offset=0, forward=True, maximum=None, check=1):
        """Use a pattern to locate frame starts near the current position.

        Note that the current position is always included.

        Parameters
        ----------
        pattern : header, ~numpy.ndaray, bytes, int, or iterable of int
            Synchronization pattern to look for.  If a header or header class,
            :meth:`~baseband.vlbi_base.header.VLBIHeaderBase.invariant_pattern`
            is used to create a masked pattern, using invariant keys from
            :meth:`~baseband.vlbi_base.header.VLBIHeaderBase.invariants`.
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
        ~baseband.vlbi_base.base.HeaderNotFoundError
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
        with self.temporary_offset():
            self.seek(0)
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


class VLBIStreamBase:
    """VLBI file wrapper, allowing access as a stream of data."""

    _sample_shape_maker = None

    def __init__(self, fh_raw, header0, sample_rate, samples_per_frame,
                 unsliced_shape, bps, complex_data, squeeze, subset=(),
                 fill_value=0., verify=True):
        self.fh_raw = fh_raw
        self._header0 = header0
        self._bps = bps
        self._complex_data = complex_data
        self.samples_per_frame = samples_per_frame
        self.sample_rate = sample_rate
        self._frame_rate = (sample_rate / samples_per_frame).to(u.Hz)
        self.offset = 0
        self._fill_value = fill_value

        if self._sample_shape_maker is not None:
            self._unsliced_shape = self._sample_shape_maker(*unsliced_shape)
        else:
            self._unsliced_shape = unsliced_shape

        self._squeeze = bool(squeeze)
        if subset is None:
            subset = ()
        elif not isinstance(subset, tuple):
            subset = (subset,)
        self._subset = subset
        self._sample_shape = self._get_sample_shape()
        self._frame_index = None

        self.verify = verify

    @property
    def squeeze(self):
        """Whether data arrays have dimensions with length unity removed.

        If `True`, data read out has such dimensions removed, and data
        passed in for writing has them inserted.
        """
        return self._squeeze

    @property
    def subset(self):
        """Specific components of the complete sample to decode.

        The order of dimensions is the same as for `sample_shape`.  Set by
        the class initializer.
        """
        return self._subset

    def _get_sample_shape(self):
        """Get shape of possibly squeezed samples."""
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

    @property
    def sample_shape(self):
        """Shape of a complete sample (possibly subset or squeezed)."""
        return self._sample_shape

    def _get_time(self, header):
        """Get time from a header."""
        # Subclasses can override this if information is needed beyond that
        # provided in the header.
        return header.time

    def _set_time(self, header, time):
        """Set time in a header."""
        # Subclasses can override this if information is needed beyond that
        # provided in the header.
        header.time = time

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

    @samples_per_frame.setter
    def samples_per_frame(self, samples_per_frame):
        try:
            self._samples_per_frame = operator.index(samples_per_frame)
        except Exception as exc:
            exc.args += ("samples per frame must have an integer value.",)
            raise exc

    @property
    def sample_rate(self):
        """Number of complete samples per second."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        # Check if sample_rate is a time rate.
        try:
            sample_rate.to(u.Hz)
        except u.UnitsError as exc:
            exc.args += ("sample rate must have units of 1 / time.",)
            raise
        self._sample_rate = sample_rate

    @property
    def verify(self):
        """Whether to do consistency checks on frames being read."""
        return self._verify

    @verify.setter
    def verify(self, verify):
        self._verify = bool(verify) if verify != 'fix' else verify

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

    def __repr__(self):
        return ("<{s.__class__.__name__} name={s.name} offset={s.offset}\n"
                "    sample_rate={s.sample_rate},"
                " samples_per_frame={s.samples_per_frame},\n"
                "    sample_shape={s.sample_shape}, bps={s.bps},\n"
                "    {sub}start_time={s.start_time.isot}>"
                .format(s=self, sub=('subset={0}, '.format(self.subset)
                                     if self.subset else '')))


class VLBIStreamReaderBase(VLBIStreamBase):

    def __init__(self, fh_raw, header0, sample_rate, samples_per_frame,
                 unsliced_shape, bps, complex_data, squeeze, subset,
                 fill_value, verify):

        if sample_rate is None:
            try:
                sample_rate = (samples_per_frame
                               * fh_raw.get_frame_rate()).to(u.MHz)

            except Exception as exc:
                exc.args += ("the sample rate could not be auto-detected. "
                             "This can happen if the file is too short to "
                             "determine the sample rate, or because it is "
                             "corrupted.  Try passing in an explicit "
                             "`sample_rate`.",)
                raise

        super().__init__(
            fh_raw, header0, sample_rate, samples_per_frame, unsliced_shape,
            bps, complex_data, squeeze, subset, fill_value, verify)

        if hasattr(header0, 'frame_nbytes'):
            self._raw_offsets = RawOffsets(frame_nbytes=header0.frame_nbytes)

    info = VLBIStreamReaderInfo()

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

    def _get_sample_shape(self):
        """Get shape by applying squeeze and subset to a dummy data sample."""
        # First apply base class, which squeezes if needed.
        sample_shape = super()._get_sample_shape()
        if not self.subset:
            return sample_shape

        # Now apply subset to a dummy sample that has the sample number as its
        # value (where 13 is to bring bad luck to over-complicated subsets).
        dummy_data = np.arange(13.)
        dummy_sample = np.rollaxis(  # Use moveaxis when numpy_min>=1.11
            (np.zeros(sample_shape)[..., np.newaxis] + dummy_data), -1)
        try:
            dummy_subset = dummy_sample[(slice(None),) + self.subset]
            # Check whether subset was in range and whether sample numbers were
            # preserved (latter should be, but check anyway).
            assert 0 not in dummy_subset.shape
            assert np.all(dummy_subset == dummy_data.reshape(
                (-1,) + (1,) * (dummy_subset.ndim - 1)))
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
        """Last header of the file."""
        with self.fh_raw.temporary_offset() as fh_raw:
            fh_raw.seek(-self.header0.frame_nbytes, 2)
            try:
                return fh_raw.find_header(self.header0, forward=False,
                                          check=(-1, 1))
            except HeaderNotFoundError as exc:
                exc.args += ("corrupt VLBI frame? No frame in last {0} bytes."
                             .format(2 * self.header0.frame_nbytes),)
                raise

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
                + (self.samples_per_frame / self.sample_rate).to(u.s))

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

    def seek(self, offset, whence=0):
        """Change the stream position.

        This works like a normal filehandle seek, but the offset is in samples
        (or a relative or absolute time).

        Parameters
        ----------
        offset : int, `~astropy.units.Quantity`, or `~astropy.time.Time`
            Offset to move to.  Can be an (integer) number of samples,
            an offset in time units, or an absolute time.
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
            self.offset = self._nsample + offset
        else:
            raise ValueError("invalid 'whence'; should be 0 or 'start', 1 or "
                             "'current', or 2 or 'end'.")

        return self.offset

    @property
    def dtype(self):
        # TODO: arguably, this should be inferred from an actual payload.
        return np.dtype(np.complex64 if self.complex_data else np.float32)

    def readable(self):
        """Whether the file can be read and decoded."""
        return self.fh_raw.readable and self.fh_raw.info.readable

    def read(self, count=None, out=None):
        """Read a number of complete (or subset) samples.

        The range retrieved can span multiple frames.

        Parameters
        ----------
        count : int or None, optional
            Number of complete/subset samples to read.  If `None` (default) or
            negative, the whole file is read.  Ignored if ``out`` is given.
        out : None or array, optional
            Array to store the data in. If given, ``count`` will be inferred
            from the first dimension; the other dimension should equal
            `sample_shape`.

        Returns
        -------
        out : `~numpy.ndarray` of float or complex
            The first dimension is sample-time, and the remainder given by
            `sample_shape`.
        """
        if out is None:
            if count is None or count < 0:
                count = self._nsample - self.offset
                if count < 0:
                    raise EOFError("cannot read from beyond end of file.")

            out = np.empty((count,) + self.sample_shape, dtype=self.dtype)
        else:
            assert out.shape[1:] == self.sample_shape, (
                "'out' must have trailing shape {}".format(self.sample_shape))
            count = out.shape[0]

        offset0 = self.offset
        sample = 0
        while sample < count:
            # For current position, get frame plus offset in that frame.
            frame_index, sample_offset = divmod(self.offset,
                                                self.samples_per_frame)
            if frame_index != self._frame_index:
                self._frame = self._read_frame(frame_index)
                self._frame.fill_value = self.fill_value
                self._frame_index = frame_index

            frame = self._frame

            nsample = min(count - sample, len(frame) - sample_offset)
            data = frame[sample_offset:sample_offset + nsample]
            data = self._squeeze_and_subset(data)
            # Copy to relevant part of output.
            out[sample:sample + nsample] = data
            sample += nsample
            # Explicitly set offset (just in case read_frame adjusts it too).
            self.offset = offset0 + sample

        return out

    _next_index = None

    def _read_frame(self, index):
        """Base implementation of reading a frame.

        This contains two pieces which subclasses can override as needed
        (or override the whole thing).
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
                frame_index = self._tell_frame(frame)
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
                self._next_index = self._tell_frame(self._next_frame)
        except Exception as exc:
            return self._bad_frame(index, frame, exc)

        return frame

    def _bad_frame(self, index, frame, exc):
        """Deal with a bad frame.

        Parameters
        ----------
        index : int
            Frame index that is to be read.
        frame : `~baseband.vlbi_base.frame.VLBIFrameBase` or None
            Frame that was read without failure.  If not `None`, either
            the frame index is wrong or the next frame could not be read.
        exc : Exception
            Exception that led to the call.
        """
        if (frame is not None and self._tell_frame(frame) == index
                and index == self._tell_frame(self._last_header)):
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
        header_index = self._tell_frame(header)
        if header_index < index:
            exc.args += (msg + ' There appears to be excess data.')
            raise exc

        # Go backward until we find previous frame, storing offsets
        # as we go.  We again increase the maximum since we may need
        # to jump over a bad bit.
        while header_index >= index:
            raw_pos = self.fh_raw.tell()
            header1 = header
            header1_index = header_index
            self.fh_raw.seek(-1, 1)
            try:
                header = self.fh_raw.find_header(
                    self.header0, forward=False,
                    maximum=4*self.header0.frame_nbytes)
            except HeaderNotFoundError:
                exc.args += (msg + ' Could not find previous index.',)
                raise exc

            header_index = self._tell_frame(header)
            # While we are at it, update the list of known indices.
            self._raw_offsets[header1_index] = raw_pos

        # Move back to position of last good header (header1).
        self.fh_raw.seek(raw_pos)

        if header1_index > index:
            # Frame is missing!
            msg += ' The frame seems to be missing.'
            # Construct a missing frame.
            header = header1.copy()
            self._set_time(header, self.time)
            frame = self._frame.__class__(header, self._frame.payload,
                                          valid=False)

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
            assert self._tell_frame(frame) == index

        warnings.warn(msg)
        return frame

    def _seek_frame(self, index):
        """Move the underlying file pointer to the frame of the given index."""
        return self.fh_raw.seek(self._raw_offsets[index])

    def _fh_raw_read_frame(self):
        """Read a frame at the current position of the underlying file."""
        return self.fh_raw.read_frame(verify=self.verify)

    def _tell_frame(self, frame):
        """Get the index of the frame relative to the first frame."""
        dt = self._get_time(frame) - self.start_time
        return int(round((dt * self._frame_rate).to_value(u.one)))


class VLBIStreamWriterBase(VLBIStreamBase):

    def __init__(self, fh_raw, header0, sample_rate, samples_per_frame,
                 unsliced_shape, bps, complex_data, squeeze, subset,
                 fill_value, verify):

        if sample_rate is None:
            raise ValueError("must pass in an explicit `sample_rate`.")

        super().__init__(
            fh_raw, header0, sample_rate, samples_per_frame, unsliced_shape,
            bps, complex_data, squeeze, subset, fill_value, verify)

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
            frame_index, sample_offset = divmod(self.offset,
                                                self.samples_per_frame)
            if frame_index != self._frame_index:
                self._frame = self._make_frame(frame_index)
                self._frame_index = frame_index
                self._valid = valid
            else:
                self._valid &= valid

            nsample = min(count - sample, len(self._frame) - sample_offset)
            sample_end = sample_offset + nsample
            self._frame[sample_offset:sample_end] = data[sample:
                                                         sample + nsample]
            if sample_end == self.samples_per_frame:
                self._write_frame(self._frame, valid=self._valid)
                # Be sure we do not reuse this frame (might also be needed
                # to write memmaps to disk).
                del self._frame
                self._frame_index = None

            sample += nsample
            # Explicitly set offset (just in case write_frame adjusts it too).
            self.offset = offset0 + sample

    def _write_frame(self, frame, valid=True):
        # Default implementation is to assume this is a frame that can write
        # the underlying binary file.
        frame.valid = valid
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


default_open_doc = """Open baseband file(s) for reading or writing.

Opened as a binary file, one gets a wrapped filehandle that adds
methods to read/write a frame.  Opened as a stream, the handle is
wrapped further, with methods such as reading and writing to the file
as if it were a stream of samples.

Parameters
----------
name : str or filehandle, or sequence of str
    File name, filehandle, or sequence of file names (see Notes).
mode : {'rb', 'wb', 'rs', or 'ws'}, optional
    Whether to open for reading or writing, and as a regular binary
    file or as a stream. Default: 'rs', for reading a stream.
**kwargs
    Additional arguments when opening the file as a stream.
"""


def make_opener(fmt, classes, doc='', append_doc=True):
    """Create a baseband file opener.

    Parameters
    ----------
    fmt : str
        Name of the baseband format.
    classes : dict
        With the file/stream reader/writer classes keyed by names equal to
        'FileReader', 'FileWriter', 'StreamReader', 'StreamWriter' prefixed by
        ``fmt``.  Typically, one will pass in ``classes=globals()``.
    doc : str, optional
        If given, used to define the docstring of the opener.
    append_doc : bool, optional
        If `True` (default), append ``doc`` to the default docstring rather
        than override it.
    """
    module = classes.get('__name__', None)
    classes = {cls_type: classes[fmt + cls_type]
               for cls_type in ('FileReader', 'FileWriter',
                                'StreamReader', 'StreamWriter')}

    def open(name, mode='rs', **kwargs):
        # If sequentialfile object, check that it's opened properly.
        if isinstance(name, sf.SequentialFileBase):
            assert (('r' in mode and name.mode == 'rb')
                    or ('w' in mode and name.mode == 'w+b')), (
                        "open only accepts sequential files opened in 'rb' "
                        "mode for reading or 'w+b' mode for writing.")

        # If passed some kind of list, open a sequentialfile object.
        if isinstance(name, (tuple, list, sf.FileNameSequencer)):
            if 'r' in mode:
                name = sf.open(name, 'rb')
            else:
                file_size = kwargs.pop('file_size', None)
                name = sf.open(name, 'w+b', file_size=file_size)

        # Select FileReader/Writer for binary, StreamReader/Writer for stream.
        if 'b' in mode:
            cls_type = 'File'
        else:
            cls_type = 'Stream'

        # Select reading or writing.  Check if ``name`` is a filehandle, and
        # open it if not.
        if 'w' in mode:
            cls_type += 'Writer'
            got_fh = hasattr(name, 'write')
            if not got_fh:
                name = io.open(name, 'w+b')
        elif 'r' in mode:
            cls_type += 'Reader'
            got_fh = hasattr(name, 'read')
            if not got_fh:
                name = io.open(name, 'rb')
        else:
            raise ValueError("only support opening {0} file for reading "
                             "or writing (mode='r' or 'w')."
                             .format(fmt))
        # Try wrapping ``name`` with file or stream reader (``name`` is a
        # binary filehandle at this point).
        try:
            return classes[cls_type](name, **kwargs)
        except Exception as exc:
            if not got_fh:
                try:
                    name.close()
                except Exception:  # pragma: no cover
                    pass
            raise exc

    # Load custom documentation for format.
    open.__doc__ = (default_open_doc.replace('baseband', fmt) + doc
                    if append_doc else doc)
    # This ensures the function becomes visible to sphinx.
    if module:
        open.__module__ = module

    return open
