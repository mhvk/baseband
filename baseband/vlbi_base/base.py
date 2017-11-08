import io
import warnings
import numpy as np
from collections import namedtuple
from astropy import units as u
from astropy.utils import lazyproperty


__all__ = ['u_sample', 'VLBIStreamBase', 'VLBIStreamReaderBase',
           'VLBIStreamWriterBase', 'make_opener']

u_sample = u.def_unit('sample', doc='One sample from a data stream')


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

    _frame_class = None
    _squeezed_shape = None

    def __init__(self, fh_raw, header0, sample_shape, bps, complex_data,
                 thread_ids, samples_per_frame, frames_per_second=None,
                 sample_rate=None, squeeze=True):
        super(VLBIStreamBase, self).__init__(fh_raw)
        self.header0 = header0
        self._sample_shape = sample_shape
        self.bps = bps
        self.complex_data = complex_data
        self.thread_ids = thread_ids
        self.samples_per_frame = samples_per_frame
        if frames_per_second is None:
            frames_per_second = sample_rate.to(u.Hz).value / samples_per_frame
            if frames_per_second % 1:  # pragma: no cover
                warnings.warn("Sampling rate {0} and samples per frame {1} "
                              "imply non-integer number of frames per "
                              "second".format(sample_rate, samples_per_frame))
            else:
                frames_per_second = int(frames_per_second)

        self.frames_per_second = frames_per_second
        self.offset = 0
        self.squeeze = squeeze

    @property
    def sample_shape(self):
        """Shape of a data sample (possibly squeezed)."""
        if self.squeeze:
            if not self._squeezed_shape:
                field_names = getattr(self._sample_shape, '_fields', None)
                sqz_dims = [dim for dim in self._sample_shape if dim > 1]
                if field_names is None:
                    self._squeezed_shape = tuple(sqz_dims)
                else:
                    sqz_names = [field for field, dim in
                                 zip(field_names, self._sample_shape) if
                                 dim > 1]
                    sqz_shp_cls = namedtuple('SampleShape',
                                             ','.join(sqz_names))
                    self._squeezed_shape = sqz_shp_cls(*sqz_dims)
            return self._squeezed_shape
        else:
            return self._sample_shape

    def _unsqueeze(self, data):
        new_shape = list(data.shape)
        for i, dim in enumerate(self._sample_shape):
            if dim == 1:
                new_shape.insert(i + 1, 1)
        return data.reshape(new_shape)

    def _get_time(self, header):
        """Get time from a header."""
        # Subclasses can override this if information is needed beyond that
        # provided in the header.
        return header.time

    @lazyproperty
    def time0(self):
        """Start time."""
        return self._get_time(self.header0)

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
            return self.time0 + self.tell(unit=u.s)

        return (self.offset * u_sample).to(unit, equivalencies=[(u.s, u.Unit(
            self.samples_per_frame * self.frames_per_second * u_sample))])

    def _frame_info(self):
        offset = (self.offset +
                  self.header0['frame_nr'] * self.samples_per_frame)
        full_frame_nr, extra = divmod(offset, self.samples_per_frame)
        dt, frame_nr = divmod(full_frame_nr, self.frames_per_second)
        return int(dt), int(frame_nr), extra

    def __repr__(self):
        return ("<{s.__class__.__name__} name={s.name} offset={s.offset}\n"
                "    frames_per_second={s.frames_per_second},"
                " samples_per_frame={s.samples_per_frame},\n"
                "    sample_shape={s.sample_shape}, bps={s.bps},\n"
                "    {t}(start) time={s.time0.isot}>"
                .format(s=self, t=('thread_ids={0}, '.format(self.thread_ids)
                                   if self.thread_ids else '')))


class VLBIStreamReaderBase(VLBIStreamBase):

    def __init__(self, fh_raw, header0, sample_shape, bps, complex_data,
                 thread_ids, samples_per_frame, frames_per_second=None,
                 sample_rate=None, squeeze=True):

        if frames_per_second is None and sample_rate is None:
            try:
                frames_per_second = self._get_frame_rate(fh_raw,
                                                         header0)
            except Exception as exc:
                exc.args += ("The frame rate could not be auto-detected. "
                             "This can happen if the file is too short to "
                             "determine the frame rate, or because it is "
                             "corrupted.  Try passing in an explicit "
                             "'frames_per_second'.",)
                raise

        super(VLBIStreamReaderBase, self).__init__(
            fh_raw, header0, sample_shape, bps, complex_data, thread_ids,
            samples_per_frame, frames_per_second, sample_rate, squeeze)

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

        The function cycles through headers, starting from the file
        pointer's current position, to find the next frame whose
        frame number is zero while keeping track of the largest frame
        number yet found.

        ``_get_frame_rate`` is called when the number of frames
        per second is not user-provided or deducable from header
        information.  If less than one second of data exists in the
        file, the function will raise an EOFError.  It also returns
        an error if any header cannot be read or does not verify as
        correct.
        """
        oldpos = fh.tell()
        header = header_template.fromfile(fh)
        frame_nr0 = header['frame_nr']
        sec0 = header.seconds
        while header['frame_nr'] == frame_nr0:
            fh.seek(header.payloadsize, 1)
            header = header_template.fromfile(fh)
        max_frame = frame_nr0
        while header['frame_nr'] > 0:
            max_frame = max(header['frame_nr'], max_frame)
            fh.seek(header.payloadsize, 1)
            header = header_template.fromfile(fh)

        if header.seconds != sec0 + 1:  # pragma: no cover
            warnings.warn("Header time changed by more than 1 second?")

        fh.seek(oldpos)
        return max_frame + 1

    @lazyproperty
    def _header1(self):
        """Last header of the file."""
        raw_offset = self.fh_raw.tell()
        self.fh_raw.seek(-self.header0.framesize, 2)
        header1 = self.find_header(template_header=self.header0,
                                   maximum=10*self.header0.framesize,
                                   forward=False)
        self.fh_raw.seek(raw_offset)
        if header1 is None:
            raise ValueError("Corrupt VLBI frame? No frame in last {0} bytes."
                             .format(10 * self.header0.framesize))
        return header1

    @lazyproperty
    def _time1(self):
        """Time of the sample just beyond the last one in the file."""
        return self._get_time(self._header1) + u.s / self.frames_per_second

    @property
    def size(self):
        """Number of samples in the file."""
        return int(round((self._time1 - self.time0).to(u.s).value *
                         self.frames_per_second * self.samples_per_frame))

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
            and from the end if ``2``.  Ignored if ``offset`` is a time.`
        """
        try:
            offset = offset.__index__()
        except Exception:
            try:
                offset = offset - self.time0
            except Exception:
                pass
            else:
                whence = 0

            offset = offset.to(u_sample, equivalencies=[(u.s, u.Unit(
                self.samples_per_frame * self.frames_per_second * u_sample))])
            offset = int(round(offset.value))

        if whence == 0:
            self.offset = offset
        elif whence == 1:
            self.offset += offset
        elif whence == 2:
            self.offset = self.size + offset
        else:
            raise ValueError("invalid 'whence'; should be 0, 1, or 2.")

        return self.offset


class VLBIStreamWriterBase(VLBIStreamBase):
    def close(self):
        extra = self.offset % self.samples_per_frame
        if extra != 0:
            warnings.warn("Closing with partial buffer remaining.  "
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
            if kwargs:
                raise TypeError('got unexpected arguments {}'
                                .format(kwargs.keys()))
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
            raise ValueError("Only support opening {0} file for reading "
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
