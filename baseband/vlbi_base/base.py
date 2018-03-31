# Licensed under the GPLv3 - see LICENSE.rst
from __future__ import division, unicode_literals, print_function

import io
import warnings
import numpy as np
from collections import namedtuple
import astropy.units as u
from astropy.utils import lazyproperty


__all__ = ['VLBIStreamBase', 'VLBIStreamReaderBase', 'VLBIStreamWriterBase',
           'make_opener']


class VLBIFileBase(object):
    """VLBI file wrapper, used to add frame methods to a binary data file.

    The underlying file is stored in ``fh`` and all attributes that do not
    exist on the class itself are looked up on it.
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

    def __repr__(self):
        return "{0}(fh_raw={1})".format(self.__class__.__name__, self.fh_raw)


class VLBIStreamBase(VLBIFileBase):
    """VLBI file wrapper, allowing access as a stream of data."""

    _sample_shape_maker = None

    def __init__(self, fh_raw, header0, unsliced_shape, bps, complex_data,
                 subset, samples_per_frame, sample_rate, fill_value=0.,
                 squeeze=True):
        super(VLBIStreamBase, self).__init__(fh_raw)
        self._header0 = header0
        self._bps = bps
        self._complex_data = complex_data
        self.samples_per_frame = samples_per_frame
        self.sample_rate = sample_rate
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

    @property
    def squeeze(self):
        """Whether data arrays have dimensions with length unity removed.

        If `True`, data read out has such dimensions removed, and data
        passed in for writing has them inserted.
        """
        return self._squeeze

    @property
    def subset(self):
        """Specific elements (threads/channels) of the sample to read.

        The order of dimensions is the same as for `sample_shape`.  Set by the
        class initializer.
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

    @lazyproperty
    def start_time(self):
        """Start time of the file.

        See also `time` for the time of the sample pointer's current offset,
        and (if available) `stop_time` for the time at the end of the file.
        """
        return self._get_time(self.header0)

    @property
    def time(self):
        """Time of the sample pointer's current offset in file.

        See also `start_time` for the start time, and (if available)
        `stop_time` for the end time, of the file.
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
        """Whether the decoded data is complex."""
        return self._complex_data

    @property
    def samples_per_frame(self):
        """Number of complete samples per frame."""
        return self._samples_per_frame

    @samples_per_frame.setter
    def samples_per_frame(self, samples_per_frame):
        try:
            self._samples_per_frame = samples_per_frame.__index__()
        except Exception:
            raise TypeError("samples per frame must have an integer value.")

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

    def tell(self, unit=None):
        """Current offset in file.

        Parameters
        ----------
        unit : `~astropy.units.Unit` or str, optional
            Time unit the offset should be returned in.  By default, no unit
            is used, i.e., an integer enumerating samples is returned. For the
            special string 'time', the absolute time is calculated.

        Returns
        -------
        offset : int, `~astropy.units.Quantity`, or `~astropy.time.Time`
             Offset in current file (or time at current position)
        """
        if unit is None:
            return self.offset

        if unit == 'time':
            return self.start_time + self.tell(unit=u.s)

        return (self.offset / self.sample_rate).to(unit)

    def __repr__(self):
        return ("<{s.__class__.__name__} name={s.name} offset={s.offset}\n"
                "    sample_rate={s.sample_rate},"
                " samples_per_frame={s.samples_per_frame},\n"
                "    sample_shape={s.sample_shape}, bps={s.bps},\n"
                "    {sub}start_time={s.start_time.isot}>"
                .format(s=self, sub=('subset={0}, '.format(self.subset)
                                     if self.subset else '')))


class VLBIStreamReaderBase(VLBIStreamBase):

    def __init__(self, fh_raw, header0, unsliced_shape, bps, complex_data,
                 subset, samples_per_frame, sample_rate=None, fill_value=0.,
                 squeeze=True):

        if sample_rate is None:
            try:
                sample_rate = (samples_per_frame *
                               self._get_frame_rate(fh_raw, header0)).to(u.MHz)

            except Exception as exc:
                exc.args += ("the sample rate could not be auto-detected. "
                             "This can happen if the file is too short to "
                             "determine the sample rate, or because it is "
                             "corrupted.  Try passing in an explicit "
                             "`sample_rate`.",)
                raise

        super(VLBIStreamReaderBase, self).__init__(
            fh_raw, header0, unsliced_shape, bps, complex_data, subset,
            samples_per_frame, sample_rate, fill_value, squeeze)

    def _squeeze_and_subset(self, data):
        """Possibly remove unit dimensions and subset the given data.

        The first dimension (sample number) is never removed.
        """
        if self.squeeze:
            data = data.reshape(data.shape[:1] +
                                tuple(sh for sh in data.shape[1:] if sh > 1))
        if self.subset:
            data = data[(slice(None),) + self.subset]

        return data

    def _get_sample_shape(self):
        """Get shape by applying squeeze and subset to a dummy data sample."""
        # First apply base class, which squeezes if needed.
        sample_shape = super(VLBIStreamReaderBase, self)._get_sample_shape()
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
        if (not hasattr(sample_shape, '_fields') or subset_shape == () or
                len(self.subset) > len(sample_shape)):
            return subset_shape

        # We can only associate names when indexing each dimension separately
        # gives a consistent result with the complete subset.
        subset_axis = 0
        fields = []
        subset = self.subset + (slice(None),) * (len(sample_shape) -
                                                 len(self.subset))
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

    @staticmethod
    def _get_frame_rate(fh, header_template):
        """Returns the number of frames per second.

        Parameters
        ----------
        fh : io.BufferedReader
            Binary file handle.
        header_template : header class or instance
            Definition or instance of file format's header class.

        Returns
        -------
        framerate : `~astropy.units.Quantity`
            Frames per second, in Hz.

        Notes
        -----

        The function cycles through headers, starting from the file pointer's
        current position, to find the next frame whose frame number is zero
        while keeping track of the largest frame number yet found.

        ``_get_frame_rate`` is called when the sample rate is not user-provided
        or deducible from header information.  If less than one second of data
        exists in the file, the function will raise an EOFError.  It also
        returns an error if any header cannot be read or does not verify as
        correct.
        """
        oldpos = fh.tell()
        header = header_template.fromfile(fh)
        frame_nr0 = header['frame_nr']
        while header['frame_nr'] == frame_nr0:
            fh.seek(header.payloadsize, 1)
            header = header_template.fromfile(fh)
        max_frame = frame_nr0
        while header['frame_nr'] > 0:
            max_frame = max(header['frame_nr'], max_frame)
            fh.seek(header.payloadsize, 1)
            header = header_template.fromfile(fh)

        fh.seek(oldpos)
        return (max_frame + 1) * u.Hz

    @lazyproperty
    def _last_header(self):
        """Last header of the file."""
        raw_offset = self.fh_raw.tell()
        self.fh_raw.seek(-self.header0.framesize, 2)
        last_header = self.find_header(forward=False,
                                       template_header=self.header0)
        self.fh_raw.seek(raw_offset)
        if last_header is None:
            raise ValueError("corrupt VLBI frame? No frame in last {0} bytes."
                             .format(10 * self.header0.framesize))
        return last_header

    @lazyproperty
    def stop_time(self):
        """Time at the end of the file, just after the last sample.

        See also `start_time` for the start time of the file, and `time` for
        the time of the sample pointer's current offset.
        """
        return (self._get_time(self._last_header) +
                (self.samples_per_frame / self.sample_rate).to(u.s))

    @property
    def size(self):
        """Number of samples in the file."""
        return int(((self.stop_time - self.start_time) *
                    self.sample_rate).to(u.one).round())

    @property
    def fill_value(self):
        """Value to use for invalid or missing data. Default: 0."""
        return self._fill_value

    def seek(self, offset, whence=0):
        """Change stream position.

        This works like a normal seek, but the offset is in samples
        (or a relative or absolute time).

        Parameters
        ----------
        offset : int, `~astropy.units.Quantity`, or `~astropy.time.Time`
            Offset to move to.  Can be an (integer) number of samples,
            an offset in time units, or an absolute time.
        whence : int
            Like regular seek, the offset is taken to be from the start if
            ``whence=0`` (default), from the current position if ``1``,
            and from the end if ``2``.  One can use ``'start'``, ``'current'``,
            or ``'end'`` for ``0``, ``1``, or ``2``, respectively.  Ignored if
            ``offset`` is a time.`
        """
        try:
            offset = offset.__index__()
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
            self.offset = self.size + offset
        else:
            raise ValueError("invalid 'whence'; should be 0 or 'start', 1 or"
                             "'current', or 2 or 'end'.")

        return self.offset

    @property
    def dtype(self):
        return np.complex64 if self.complex_data else np.float32

    def read(self, count=None, out=None):
        """Read a number of complete (or subset) samples.

        The range retrieved can span multiple frames.

        Parameters
        ----------
        count : int
            Number of complete/subset samples to read.  If omitted or negative,
            the whole file is read.  Ignored if ``out`` is given.
        out : `None` or array
            Array to store the data in. If given, ``count`` will be inferred
            from the first dimension.  The other dimension should equal
            ``sample_shape``.

        Returns
        -------
        out : array of float or complex
            The first dimension is sample-time, and the second, given by
            ``sample_shape``, is (channel,).  Any dimension of length unity is
            removed if ``self.squeeze=True``.
        """
        if out is None:
            if count is None or count < 0:
                count = self.size - self.offset
                if count < 0:
                    raise EOFError("cannot read from beyond end of file.")

            out = np.empty((count,) + self.sample_shape, dtype=self.dtype)
        else:
            assert out.shape[1:] == self.sample_shape, (
                "'out' should have trailing shape {}".format(self.sample_shape))
            count = out.shape[0]

        offset0 = self.offset
        sample = 0
        while count > 0:
            # For current position, get frame plus offset in that frame.
            frame_index, sample_offset = divmod(self.offset,
                                                self.samples_per_frame)
            if frame_index != self._frame_index:
                self._frame = self._read_frame(frame_index)
                self._frame_index = frame_index

            frame = self._frame
            nsample = min(count, len(frame) - sample_offset)
            data = frame[sample_offset:sample_offset + nsample]
            data = self._squeeze_and_subset(data)
            # Copy to relevant part of output.
            out[sample:sample + nsample] = data
            sample += nsample
            # Explicitly set offset (just in case read_frame adjusts it too).
            self.offset = offset0 + sample
            count -= nsample

        return out


class VLBIStreamWriterBase(VLBIStreamBase):

    def _unsqueeze(self, data):
        new_shape = list(data.shape)
        for i, dim in enumerate(self._unsliced_shape):
            if dim == 1:
                new_shape.insert(i + 1, 1)
        return data.reshape(new_shape)

    def write(self, data, invalid_data=False):
        """Write data, buffering by frames as needed.

        Parameters
        ----------
        data : array
            Piece of data to be written, with sample dimensions as given by
            ``sample_shape``. This should be properly scaled to make best use
            of the dynamic range delivered by the encoding.
        invalid_data : bool, optional
            Whether the current data is invalid.  Defaults to `False`.
        """
        assert data.shape[1:] == self.sample_shape, (
            "'data' should have trailing shape {}".format(self.sample_shape))

        if self.squeeze:
            data = self._unsqueeze(data)

        count = data.shape[0]
        offset0 = self.offset
        sample = 0
        while count > 0:
            frame_index, sample_offset = divmod(self.offset,
                                                self.samples_per_frame)
            if frame_index != self._frame_index:
                self._frame = self._make_frame(frame_index)
                self._frame_index = frame_index
                self._valid = not invalid_data
            else:
                self._valid &= not invalid_data

            nsample = min(count, len(self._frame) - sample_offset)
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
            count -= nsample

    def _write_frame(self, frame, valid=True):
        # default implementation is to assume this is a frame and use
        # the binary file writer.
        frame.valid = valid
        self.write_frame(frame)

    def close(self):
        extra = self.offset % self.samples_per_frame
        if extra != 0:
            warnings.warn("closing with partial buffer remaining.  "
                          "Writing padded frame, marked as invalid.")
            self.write(np.zeros((self.samples_per_frame - extra,) +
                                self.sample_shape), invalid_data=True)
            assert self.offset % self.samples_per_frame == 0
        return super(VLBIStreamWriterBase, self).close()


default_open_doc = """Open baseband file for reading or writing.

Opened as a binary file, one gets a wrapped file handle that adds
methods to read/write a frame.  Opened as a stream, the handle is
wrapped further, with methods such as read and write access the file
as if it were a stream of samples.

Parameters
----------
name : str or filehandle
    File name or handle
mode : {'rb', 'wb', 'rs', or 'ws'}, optional
    Whether to open for reading or writing, and as a regular binary
    file or as a stream (default is reading a stream).
**kwargs
    Additional arguments when opening the file as a stream.
"""


def make_opener(fmt, classes, doc='', append_doc=True):
    """Create a baseband file opener.

    Parameters
    ----------
    fmt : str
        Name of the baseband format
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
        if 'b' in mode:
            cls_type = 'File'
        else:
            cls_type = 'Stream'

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
        try:
            return classes[cls_type](name, **kwargs)
        except Exception as exc:
            if not got_fh:
                try:
                    name.close()
                except Exception:  # pragma: no cover
                    pass
            raise exc

    open.__doc__ = (default_open_doc.replace('baseband', fmt) + doc
                    if append_doc else doc)
    # This ensures the function becomes visible to sphinx.
    if module:
        open.__module__ = module

    return open
