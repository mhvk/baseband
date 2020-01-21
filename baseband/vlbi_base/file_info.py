# Licensed under the GPLv3 - see LICENSE
"""Provide a base class for "info" properties.

Loosely based on `~astropy.utils.data_info.DataInfo`.
"""
import copy
import operator
import warnings
from collections import OrderedDict

import numpy as np
from astropy import units as u
from astropy.time import Time


__all__ = ['info_item', 'VLBIInfoMeta', 'VLBIInfoBase',
           'VLBIFileReaderInfo', 'VLBIStreamReaderInfo']


class info_item:
    """Like a lazy property, evaluated only once.

    Can be used as a decorator.

    It replaces itself with the evaluation of the function, i.e.,
    it is not a data descriptor.

    Any errors encountered during the evaluation are stored in the
    instances ``errors`` dict.

    Parameters
    ----------
    attr : str or callable, optional
        If a string, assumes we will get that attribute from ``needs``,
        otherwise the attr will be called to calculate the value.  In
        this case, the name of the attribute will be taken from the
        callable's name.  If this argument is not given, it is assumed
        the class is used as a decorator, and a wrapper is returned.
    needs : str or tuple of str
        The attributes that need to be present to get or calculate
        ``attr``.  If ``attr`` is a string, this should be where the
        attribute should be gotten from (e.g., 'header0' or '_parent');
        if not given, the attribute will simply be set to ``default``.
    default : value, optional
        The value to return if the needs are not met.  Default: `None`.
    doc : str, optional
        Docstring of the descriptor.  If not given will be taken from
        ``attr`` if a function, otherwise constructed.  (Hard to access
        from within python, but useful for sphinx documentation.)
    missing : str, optional
        If the value could be calculated or retrieved, but is `None`,
        then add this string to the ``missing`` attribute of the instance.
        Used, e.g., for Mark 5B to give a helpful message if ``bps``
        is not found on the file reader instance.
    copy : bool
        Whether the copy the value if it is retrieved.  This can be
        useful, e.g., if the value is expected to be a `dict` and an
        independent copy should be made.
    """
    def __new__(cls, attr=None, needs=(), default=None, doc=None,
                missing=None, copy=False):
        if attr is None:
            def wrapper(func):
                return cls(func, needs=needs, default=default, doc=doc,
                           missing=missing, copy=copy)

            return wrapper

        return super().__new__(cls)

    def __init__(self, attr, needs=(), default=None, doc=None,
                 missing=None, copy=False):
        needs = tuple(needs) if isinstance(needs, (tuple, list)) else (needs,)
        if callable(attr):
            self.fget = attr
            self.attr = attr.__name__
            if doc is None:
                doc = attr.__doc__
        else:
            self.attr = attr
            full_attr = '.'.join(needs+(attr,))
            self.fget = operator.attrgetter(full_attr) if needs else None
            if doc is None:
                doc = "Link to " + full_attr.replace('_parent', 'parent')

        self.needs = needs if '_parent' in needs else ('_parent',) + needs
        self.default = default
        self.missing = missing
        self.copy = copy
        self.__doc__ = doc

    def __get__(self, instance, cls=None):
        if instance is None:
            return self

        if self.fget and all(getattr(instance, need, None) is not None
                             for need in self.needs):
            try:
                value = self.fget(instance)
            except Exception as exc:
                instance.errors[self.attr] = exc
                value = self.default
            else:
                if value is None:
                    if self.missing:
                        instance.missing[self.attr] = self.missing
                    value = self.default

        else:
            value = self.default

        if self.copy:
            value = copy.copy(value)

        setattr(instance, self.attr, value)
        return value


class VLBIInfoMeta(type):
    # Set any default attributes according to where they are mentioned
    # (if not explicitly defined already).
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        defs = set(dct.keys())
        for attr in set(dct.get('_header0_attrs', ())) - defs:
            setattr(cls, attr, info_item(attr, needs='header0'))
        for attr in set(dct.get('_parent_attrs', ())) - defs:
            setattr(cls, attr, info_item(attr, needs='_parent'))


class VLBIInfoBase(metaclass=VLBIInfoMeta):
    """Container providing a standardized interface to file information.

    In order to ensure that information is always returned, all access
    to the parent should be via `~baseband.vlbi_base.file_info.info_item`,
    which ensures that any errors are stored in ``self.errors``.
    In addition, it may be useful to capture warnings and store them in
    ``self.warnings``.

    The instance evaluates as `True` if the underlying file is of the right
    format, and can thus, at least in principle, be read (though more
    information may be needed, given in ``missing``, or the file may be
    corrupt futher on).

    Parameters
    ----------
    parent : instance, optional
        Instance of the file or stream reader the ``info`` instance is
        attached too.  `None` if it is the class version.
    """

    attr_names = ()
    """Attributes that the container provides."""

    _parent_attrs = ()
    _parent = None

    def __init__(self, parent=None):
        if parent is not None:
            self._parent = parent
            for attr in self.attr_names:
                getattr(self, attr)

    def _up_to_date(self):
        """Determine whether the information we have stored is up to date."""
        return all(getattr(self, attr) == getattr(self._parent, attr, None)
                   for attr in self._parent_attrs)

    def __get__(self, instance, owner_cls):
        if instance is None:
            # Unbound descriptor, nothing to do.
            return self

        # Check if we have a stored and up to date copy.
        info = instance.__dict__.get('info')
        if info is None or not info._up_to_date():
            # If not, create a new instance and fill it.  Notes:
            # - We cannot change "self", as this was created on the class.
            # - We start from scratch rather than determine what is no longer
            #   up to date, since we cannot know what an update may influence
            #   (e.g., for Mark 4, a change in ref_time affect start_time).
            info = instance.__dict__['info'] = self.__class__(parent=instance)

        return info

    def __delete__(self, instance):
        # We need to define either __set__ or __delete__ since this ensures we
        # are treated as a "data descriptor", i.e., that our __get__ will get
        # called even if "info" is present in instance.__dict__; see
        # https://docs.python.org/3/howto/descriptor.html
        # __delete__ is more useful for us.
        instance.__dict__.pop('info', None)

    def __bool__(self):
        return self.format is not None

    def __call__(self):
        """Create a dict with file information.

        This includes information about checks done, possible missing
        information, as well as possible warnings and errors.
        """
        info = {}
        for attr in self.attr_names:
            value = getattr(self, attr)
            if not (value is None or (isinstance(value, dict)
                                      and value == {})):
                info[attr] = value

        return info

    def __repr__(self):
        # Use the repr for display of file information.
        if self._parent is None:
            return super().__repr__()

        if not self:
            if self._parent.closed:
                return 'File closed. Not parsable.'
            else:
                return 'Not parsable. Wrong format?'

        result = ''
        for attr in self.attr_names:
            value = getattr(self, attr)
            if isinstance(value, dict):
                prefix = '\n{}: '.format(attr)
                if attr == 'missing':
                    for msg in sorted(set(self.missing.values())):
                        keys = sorted(set(key for key in self.missing
                                          if self.missing[key] == msg))
                        result += "{} {}: {}\n".format(prefix,
                                                       ', '.join(keys), msg)
                        prefix = ' ' * (len(attr) + 2)
                else:
                    for key, val in value.items():
                        str_val = str(val) or repr(val)
                        result += "{} {}: {}\n".format(prefix, key, str_val)
                        prefix = ' ' * (len(attr) + 2)

            elif value is not None:
                if isinstance(value, Time):
                    value = Time(value, format='isot', precision=9)
                elif attr == 'sample_rate':
                    value = value.to(u.MHz)
                result += '{} = {}\n'.format(attr, value)

        return result


class VLBIFileReaderInfo(VLBIInfoBase):
    """Standardized information on file readers.

    The ``info`` descriptor has a number of standard attributes, which are
    determined from arguments passed in opening the file, from the first header
    (``info.header0``) and from possibly scanning the file to determine the
    duration of frames.

    Attributes
    ----------
    format : str or `None`
        File format, or `None` if the underlying file cannot be parsed.
    number_of_frames : int
        Number of frames in the file.
    frame_rate : `~astropy.units.Quantity`
        Number of data frames per unit of time.
    sample_rate : `~astropy.units.Quantity`
        Complete samples per unit of time.
    samples_per_frame : int
        Number of complete samples in each frame.
    sample_shape : tuple
        Dimensions of each complete sample (e.g., ``(nchan,)``).
    bps : int
        Number of bits used to encode each elementary sample.
    complex_data : bool
        Whether the data are complex.
    start_time : `~astropy.time.Time`
        Time of the first complete sample.
    readable : bool
        Whether the first sample could be read and decoded.
    missing : dict
        Entries are keyed by names of arguments that should be passed to
        the file reader to obtain full information. The associated entries
        explain why these arguments are needed.
    checks : dict
        Checks that were done to determine whether the file was readable
        (normally the only entry is 'decodable').
    errors : dict
        Any exceptions raised while trying to determine attributes or doing
        checks.  Keyed by the attributes/checks.
    warnings : dict
        Any warnings about the attributes or about the checks.
        Keyed by the attributes/checks.

    Examples
    --------
    The most common use is simply to print information::

        >>> from baseband.data import SAMPLE_MARK5B
        >>> from baseband import mark5b
        >>> fh = mark5b.open(SAMPLE_MARK5B, 'rb')
        >>> fh.info
        File information:
        format = mark5b
        number_of_frames = 4
        frame_rate = 6400.0 Hz
        bps = 2
        complex_data = False
        readable = False
        <BLANKLINE>
        missing:  nchan: needed to determine sample shape, frame rate, ...
                  kday, ref_time: needed to infer full times.

        >>> fh.close()

        >>> fh = mark5b.open(SAMPLE_MARK5B, 'rb', kday=56000, nchan=8)
        >>> fh.info
        File information:
        format = mark5b
        number_of_frames = 4
        frame_rate = 6400.0 Hz
        sample_rate = 32.0 MHz
        samples_per_frame = 5000
        sample_shape = (8,)
        bps = 2
        complex_data = False
        start_time = 2014-06-13T05:30:01.000000000
        readable = True
        <BLANKLINE>
        checks:  decodable: True
        >>> fh.close()
    """
    attr_names = ('format', 'number_of_frames', 'frame_rate', 'sample_rate',
                  'samples_per_frame', 'sample_shape', 'bps', 'complex_data',
                  'start_time', 'readable',
                  'missing', 'checks', 'errors', 'warnings')
    """Attributes that the container provides."""

    _header0_attrs = ('bps', 'complex_data', 'samples_per_frame',
                      'sample_shape')

    missing = info_item('missing', default=OrderedDict(), copy=True)
    checks = info_item('checks', default=OrderedDict(), copy=True)
    errors = info_item('errors', default=OrderedDict(), copy=True)
    warnings = info_item('warnings', default=OrderedDict(), copy=True)

    @info_item
    def header0(self):
        """Header of the first frame in the file."""
        # Here, we do not even know whether the file is open or whether we
        # have the right format. We thus use a try/except and filter out all
        # warnings.
        with self._parent.temporary_offset() as fh:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fh.seek(0)
                return fh.read_header()

    @info_item(needs='header0')
    def frame0(self):
        """First frame from the file."""
        # Try reading a frame.  This has no business failing if a
        # frame rate could be determined, but try anyway; maybe file is closed.
        with self._parent.temporary_offset() as fh:
            fh.seek(0)
            return fh.read_frame()

    @info_item(needs='frame0', default=False)
    def decodable(self):
        """Whether decoding the first frame worked."""
        # Getting the first sample can fail if we don't have the right decoder.
        first_sample = self.frame0[0]

        if not isinstance(first_sample, np.ndarray):
            raise TypeError('first sample is not an ndarray')

        return True

    @info_item(needs='header0')
    def format(self):
        """The file format."""
        return self._parent.__class__.__name__.split('File')[0].lower()

    @info_item(needs='header0')
    def frame_rate(self):
        """Number of frames per unit time."""
        return self._parent.get_frame_rate()

    @info_item(needs='header0')
    def number_of_frames(self):
        """Total number of frames."""
        with self._parent.temporary_offset() as fh:
            number_of_frames = fh.seek(0, 2) / self.header0.frame_nbytes

        if number_of_frames % 1 == 0:
            return int(number_of_frames)
        else:
            self.warnings['number_of_frames'] = (
                'file contains non-integer number ({}) of frames'
                .format(number_of_frames))
            return None

    @info_item(needs='header0')
    def start_time(self):
        """Time of the first sample."""
        return self.header0.time

    @info_item(needs='frame0', default=False)
    def readable(self):
        """Whether the file is readable and decodable."""
        self.checks['decodable'] = self.decodable
        return all(bool(v) for v in self.checks.values())

    @info_item(needs=('frame_rate', 'samples_per_frame'))
    def sample_rate(self):
        """Rate of complete samples per unit time."""
        return self.frame_rate * self.samples_per_frame

    def __repr__(self):
        result = 'File information:\n'
        result += super().__repr__()
        return result


class VLBIStreamReaderInfo(VLBIInfoBase):
    """Standardized information on stream readers.

    The ``info`` descriptor provides a few standard attributes, most of which
    can also be accessed directly on the stream filehandle, and tests basic
    readability of the stream. More detailed information on the underlying
    file is stored in its info, accessible via ``info.file_info`` (and shown
    by ``__repr__``).

    Attributes
    ----------
    start_time : `~astropy.time.Time`
        Time of the first complete sample.
    stop_time : `~astropy.time.Time`
        Time of the complete sample just beyond the end of the file.
    sample_rate : `~astropy.units.Quantity`
        Complete samples per unit of time.
    shape : tuple
        Equivalent shape of the whole file, i.e., combining the number of
        complete samples and the shape of those samples.
    bps : int
        Number of bits used to encode each elementary sample.
    complex_data : bool
        Whether the data are complex.
    verify : bool or str
        The type of verification done by the stream reader.
    readable : bool
        Whether the first and last samples could be read and decoded.
    checks : dict
        Checks that were done to determine whether the file was readable
        (normally 'continuous' and 'decodable').
    errors : dict
        Any exceptions raised while trying to determine attributes or doing
        checks.  Keyed by the attributes/checks.
    warnings : dict
        Any warnings about the attributes or about the checks.
        Keyed by the attributes/checks.
    """
    attr_names = ('start_time', 'stop_time', 'sample_rate', 'shape',
                  'format', 'bps', 'complex_data', 'verify', 'readable',
                  'checks', 'errors', 'warnings')
    """Attributes that the container provides."""

    _parent_attrs = tuple(attr for attr in attr_names
                          if attr not in ('format', 'readable'))

    # Note that we cannot have missing information in streams;
    # they cannot be opened without it.

    checks = info_item('checks', needs='file_info', copy=True,
                       default=OrderedDict())
    errors = info_item('errors', needs='file_info', copy=True,
                       default=OrderedDict())
    warnings = info_item('warnings', needs='file_info', copy=True,
                         default=OrderedDict())

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        if parent is not None:
            # Remove errors from file_info if we actually got the item.
            # (e.g., start_time if frame_rate couldn't be calculated.)
            for key in self.errors:
                if (key in self.file_info.errors
                        and getattr(self, key, None) is not None):
                    del self.errors[key]

    @info_item
    def file_info(self):
        """Information from the underlying file reader."""
        return self._parent.fh_raw.info

    @info_item(needs='file_info')
    def format(self):
        """Format of the underlying file."""
        return self.file_info.format

    @info_item
    def continuous(self):
        """Check the stream is continuous.

        Tries reading the very end.  If there is a problem, will bisect
        to find the exact offset at which the problem occurs.

        Errors are raised only to the extent verification is done.  Hence,
        if the stream was opened with ``verify=False``, many fewer problems
        will be found, while if it was opened with ``verify='fix'``, then
        for file types that support it, one will get warnings rather than
        exceptions (if the errors are fixable, of course).
        """
        fh = self._parent
        current_offset = fh.tell()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                good = -1
                bad = None
                frame = fh._tell_frame(fh._last_header)
                while frame > good:
                    try:
                        fh.seek(frame * fh.samples_per_frame)
                        fh.read(1)
                    except Exception as exc:
                        # If we know this is the right one, raise,
                        # otherwise start bisection.
                        if frame == good + 1:
                            msg = 'While reading at {}: '.format(fh.tell())
                            if isinstance(exc, UserWarning):
                                self.warnings['continuous'] = msg + str(exc)
                                return 'fixable gaps'
                            else:
                                self.errors['continuous'] = msg + repr(exc)
                                return False
                        bad = frame
                    else:
                        good = frame

                    if bad is not None:
                        # +1 to ensure we get round towards bad if needed.
                        frame = (bad + good + 1) // 2

            return 'no obvious gaps'

        finally:
            fh.seek(current_offset)

    @info_item
    def readable(self):
        """Whether the stream can be read (possibly fixing errors)."""
        if self.file_info.readable:
            self.checks['continuous'] = self.continuous
            return all(bool(v) for v in self.checks.values())
        else:
            return False

    def _up_to_date(self):
        # Stream readers can only change in how they verify.
        return self.verify == self._parent.verify

    def __call__(self):
        """Create a dict with information about the stream and the raw file."""
        info = super().__call__()
        info['file_info'] = self.file_info()
        return info

    def __repr__(self):
        result = 'Stream information:\n'
        result += super().__repr__()
        file_info = getattr(self, 'file_info', None)
        if file_info is not None:
            # Add information from the raw file, but skip atttributes and
            # extra info dicts that have been included already.
            raw_attrs = file_info.attr_names
            try:
                file_info.attr_names = [attr for attr in raw_attrs
                                        if attr not in self.attr_names]
                result += '\n' + repr(file_info)
            finally:
                file_info.attr_names = raw_attrs

        return result
