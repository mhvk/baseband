# Licensed under the GPLv3 - see LICENSE
"""Provide a base class for "info" properties.

Loosely based on `~astropy.utils.data_info.DataInfo`.
"""
import copy
import operator
import warnings

from astropy import units as u
from astropy.time import Time


__all__ = ['info_item', 'InfoBase',
           'FileReaderInfo', 'StreamReaderInfo', 'NoInfo']


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
        If a string, assumes we will get that attribute from ``needs``.
        If a callable, it will be called with the instance as its
        argument to calculate the value (i.e., it will behave like a
        property). If ``attr`` is not given, its is set after the fact
        by applying the instance to a function (i.e., using it as a
        decorator), or by defining it as an attribute of a class.
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
    _fget = None

    def __init__(self, attr=None, *, needs=(), default=None, doc=None,
                 missing=None, copy=False):
        needs = tuple(needs) if isinstance(needs, (tuple, list)) else (needs,)
        self.needs = needs
        self.default = default
        self.missing = missing
        self.copy = copy
        # attr will be a callable here if item_info is used as a decorator
        # without any arguments. For backwards compatibility, it can also
        # be the name of the attribute although that will more typically
        # pass through __set_name__.  It can still be used to let the name
        # of the info_item be different from the attribute that is gotten;
        # e.g., start_time = info_item('time', needs='header0').
        self._init_wrapup(attr, doc)

    def _init_wrapup(self, attr, doc=None):
        # Finish initialization, or update from __set_name__ or __call__
        if callable(attr):
            self._fget = attr
            self.name = attr.__name__
            doc = attr.__doc__
        elif attr is not None:
            self.name = attr
            if self._fget is None and self.needs:
                full_attr = '.'.join(self.needs+(attr,))
                self._fget = operator.attrgetter(full_attr)
                doc = "Link to " + full_attr.replace('_parent', 'parent')
        if doc and self.__doc__ is self.__class__.__doc__:
            self.__doc__ = doc

    def __set_name__(self, owner, name):
        # This call-back is entered during class definition time,
        # when the instance has been assigned to a class attribute.
        # This will define the name under which the result gets stored.
        # Will normally be the wrapped function or attribute name.
        self._init_wrapup(name)

    def __call__(self, func):
        """For use as a decorator when not yet fully initialized."""
        # We get here if info_item is used as a decorator but with other
        # arguments defined, e.g., as in @info_item(needs='header0')
        if hasattr(self, 'name'):
            raise TypeError(f"assigned {self.__class__.__name__!r}"
                            f"is not callable")
        self._init_wrapup(func)
        return self

    def __get__(self, instance, cls=None):
        if instance is None:
            return self

        if self._fget and all(getattr(instance, need, None) is not None
                              for need in self.needs):
            try:
                value = self._fget(instance)
            except Exception as exc:
                instance.errors[self.name] = exc
                value = self.default
            else:
                if value is None:
                    if self.missing:
                        instance.missing[self.name] = self.missing
                    value = self.default

        else:
            value = self.default

        if self.copy:
            value = copy.copy(value)

        setattr(instance, self.name, value)
        return value

    def __str__(self):
        short_doc = self.__doc__.split('\n')[0]
        return f"{self.name}: {short_doc}"

    def __repr__(self):
        name = self.__class__.__name__
        extra = {a: getattr(self, a) for a in
                 ('needs', 'default', 'missing', 'copy')}
        extra = ', '.join([f"{a}={v}" for a, v in extra.items()
                           if v or a == 'default' and v is not None])
        if extra:
            extra = f"\n{' '*len(name)}  {extra}"
        return f"<{name} {str(self)}{extra}>"


class InfoBase:
    """Container providing a standardized interface to file information.

    In order to ensure that information is always returned, all access
    to the parent should be via `~baseband.base.file_info.info_item`,
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

    _parent = None
    closed = info_item(needs='_parent', doc='Whether parent is closed')

    def __init__(self, parent=None):
        if parent is not None:
            self._parent = parent
            if not self.closed:
                for attr in self.attr_names:
                    getattr(self, attr)

    def _up_to_date(self):
        """Determine whether the information we have stored is up to date."""
        if not hasattr(self, '_parent_attrs'):
            # Set it on the class since it cannot change.
            cls = self.__class__
            cls._parent_attrs = tuple(
                attr for attr in dir(cls)
                if not attr.startswith('_')
                and getattr(getattr(cls, attr), 'needs', ()) == ('_parent',))

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
            return '\n'.join(
                [f"{self.__class__.__name__} (unbound) with attributes:"]
                + [f"  {getattr(self.__class__, attr)}"
                   for attr in self.attr_names])

        if self.closed:
            return "File closed. Not parsable."

        result = [self._parent.__class__.__name__.replace('Reader', '')
                  + ' information:']
        for attr in self.attr_names:
            value = getattr(self, attr)
            if isinstance(value, dict):
                prefix = f"\n{attr}: "
                spaces = ' ' * (len(attr)+2)
                if attr == 'missing':
                    for msg in sorted(set(self.missing.values())):
                        keys = sorted(set(key for key in self.missing
                                          if self.missing[key] == msg))
                        result.append(f"{prefix} {', '.join(keys)}: {msg}")
                        prefix = spaces
                else:
                    for key, val in value.items():
                        str_val = str(val) or repr(val)
                        result.append(f"{prefix} {key}: {str_val}")
                        prefix = spaces

            elif value is not None:
                if isinstance(value, Time):
                    value = Time(value, format='isot', precision=9)
                elif attr == 'sample_rate':
                    value = value.to(u.MHz)
                result.append(f"{attr} = {value}")

        if not self:
            result.append('\nNot parsable. Wrong format?')

        return '\n'.join(result)


class FileReaderInfo(InfoBase):
    """Standardized information on file readers.

    The ``info`` descriptor has a number of standard attributes, which are
    determined from arguments passed in opening the file, from the first header
    (``info.header0``) and from possibly scanning the file to determine the
    duration of frames.

    Examples
    --------
    The most common use is simply to print information::

        >>> from baseband.data import SAMPLE_MARK5B
        >>> from baseband import mark5b
        >>> fh = mark5b.open(SAMPLE_MARK5B, 'rb')
        >>> fh.info
        Mark5BFile information:
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
        Mark5BFile information:
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

    samples_per_frame = info_item(needs='header0', doc=(
        'Number of complete samples in each frame.'))
    sample_shape = info_item(needs='header0', doc=(
        'Shape of each complete sample (e.g., ``(nchan,)``).'))
    bps = info_item(needs='header0', doc=(
        'Number of bits used to encode each elementary sample.'))
    complex_data = info_item(needs='header0', doc=(
        'Whether the data are complex.'))
    start_time = info_item('time', needs='header0', doc=(
        "Time of the first sample."))

    missing = info_item(default={}, copy=True,
                        doc='dict of missing attributes.')
    checks = info_item(default={}, copy=True,
                       doc='dict of checks for readability.')
    errors = info_item(default={}, copy=True,
                       doc='dict of attributes that raised errors.')
    warnings = info_item(default={}, copy=True,
                         doc='dict of attributes that gave warnings.')

    @info_item
    def header0(self):
        """Header of the first frame in the file."""
        # Here, we do not even know whether the file is open or whether we
        # have the right format. We thus use a try/except and filter out all
        # warnings.
        with self._parent.temporary_offset(0) as fh:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return fh.read_header()

    @info_item(needs='header0')
    def frame0(self):
        """First frame from the file."""
        # Try reading a frame.  This has no business failing if a
        # frame rate could be determined, but try anyway; maybe file is closed.
        with self._parent.temporary_offset(0) as fh:
            return fh.read_frame()

    @info_item(needs='frame0', default=False)
    def decodable(self):
        """Whether decoding the first frame worked."""
        # Getting the first sample can fail if we don't have the right decoder.
        self.frame0[0]
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
            file_size = fh.seek(0, 2)

        number_of_frames = file_size / self.header0.frame_nbytes
        if number_of_frames % 1 == 0:
            return int(number_of_frames)
        else:
            self.warnings['number_of_frames'] = (
                f"file contains non-integer number "
                f"({number_of_frames}) of frames")
            return None

    @info_item(needs='frame0', default=False)
    def readable(self):
        """Whether the file is readable and decodable."""
        self.checks['decodable'] = self.decodable
        return all(bool(v) for v in self.checks.values())

    @info_item(needs=('frame_rate', 'samples_per_frame'))
    def sample_rate(self):
        """Rate of complete samples per unit time."""
        return self.frame_rate * self.samples_per_frame


class StreamReaderInfo(InfoBase):
    """Standardized information on stream readers.

    The ``info`` descriptor provides a few standard attributes, most of which
    can also be accessed directly on the stream filehandle, and tests basic
    readability of the stream. More detailed information on the underlying
    file is stored in its info, accessible via ``info.file_info`` (and shown
    by ``__repr__``).
    """
    attr_names = ('start_time', 'stop_time', 'sample_rate', 'shape',
                  'format', 'bps', 'complex_data', 'verify', 'readable',
                  'checks', 'errors', 'warnings')
    """Attributes that the container provides."""

    start_time = info_item(needs='_parent', doc=(
        'Time of the first complete sample.'))
    stop_time = info_item(needs='_parent', doc=(
        'Time of the sample just beyond the end of the file.'))
    sample_rate = info_item(needs='_parent', doc=(
        'Complete samples per unit of time.'))
    shape = info_item(needs='_parent', doc=(
        'Equivalent shape of the whole file.'))
    bps = info_item(needs='_parent', doc=(
        'Number of bits used to encode each elementary sample.'))
    complex_data = info_item(needs='_parent', doc=(
        'Whether the data are complex.'))
    verify = info_item(needs='_parent', doc=(
        'The type of verification done by the stream reader.'))

    # Note that we cannot have missing information in streams;
    # they cannot be opened without it.

    checks = info_item(needs='file_info', copy=True, default={},
                       doc='dict of checks for readability.')
    errors = info_item(needs='file_info', copy=True, default={},
                       doc='dict of attributes that raised errors')
    warnings = info_item(needs='file_info', copy=True, default={},
                         doc='dict of attributes that gave warnings')

    @info_item
    def file_info(self):
        """Information from the underlying file reader."""
        # Our regular baseband readers always have info on their raw
        # files, but we should not presume this is the case. E.g., hdf5
        # readers have an h5File underneath, without info.
        return getattr(getattr(self._parent, 'fh_raw', None), 'info', None)

    @info_item
    def format(self):
        """Format of the underlying file."""
        if self.file_info is not None:
            return self.file_info.format
        elif self.continuous is not None:
            return self._parent.__class__.__name__.split('Stream')[0].lower()

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
                frame = fh._get_index(fh._last_header)
                while frame > good:
                    try:
                        fh.seek(frame * fh.samples_per_frame)
                        fh.read(1)
                    except Exception as exc:
                        # If we know this is the right one, raise,
                        # otherwise start bisection.
                        if frame == good + 1:
                            msg = f"While reading at {fh.tell()}: "
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
        if self.file_info is not None and not self.file_info.readable:
            return False

        self.checks['continuous'] = self.continuous
        return all(bool(v) for v in self.checks.values())

    def _up_to_date(self):
        # Beyond open/close, stream readers can only change in how they verify.
        return (self.verify == self._parent.verify
                and self.closed == self._parent.closed)

    def __call__(self):
        """Create a dict with information about the stream and the raw file."""
        info = super().__call__()
        if self.file_info:
            info['file_info'] = self.file_info()
        return info

    def __repr__(self):
        result = super().__repr__()
        if self._parent is None:
            return result

        file_info = getattr(self, 'file_info', None)
        if file_info is not None:
            # Add information from the raw file, but skip atttributes and
            # extra info dicts that have been included already.
            raw_attrs = file_info.attr_names
            try:
                file_info.attr_names = [attr for attr in raw_attrs
                                        if attr not in self.attr_names]
                result += '\n\n' + repr(file_info)
            finally:
                file_info.attr_names = raw_attrs

        return result


class NoInfo:
    """Info class for cases where no useful information was returned.

    Any instance evaluates as `False`, to indicate a file for which
    the information is given is not readable.

    Parameters
    ----------
    info : str
        Information that will be displayed using ``repr``.
    """
    def __init__(self, info=None):
        self.info = info

    def __bool__(self):
        return False

    def __repr__(self):
        return f"No Info: {self.info}"
