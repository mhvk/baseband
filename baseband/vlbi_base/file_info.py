# Licensed under the GPLv3 - see LICENSE
"""Provide a base class for "info" properties.

Loosely based on `~astropy.utils.data_info.DataInfo`.
"""
import warnings
from collections import OrderedDict

import numpy as np
from astropy import units as u
from astropy.time import Time


__all__ = ['VLBIInfoMeta', 'VLBIInfoBase',
           'VLBIFileReaderInfo', 'VLBIStreamReaderInfo']


class info_property:
    """Like a property, but evaluate only once, and store errors.

    It is not a data property and replaces itself with the evaluation
    of the function.
    """
    def __new__(cls, fget=None, name=None, needs=(), default=None):
        if fget is None:
            def wrapper(func):
                return cls(func, name=name, needs=needs, default=default)

            return wrapper

        return super().__new__(cls)

    def __init__(self, fget, name=None, needs=(), default=None):
        self.fget = fget
        self.name = fget.__name__ if name is None else name
        self.method = name is None
        self.needs = needs if isinstance(needs, (tuple, list)) else (needs,)
        self.default = default

    def __get__(self, instance, cls=None):
        if instance is None:
            return self

        if all(getattr(instance, need, None) is not None
               for need in self.needs):
            args = (instance,) if self.method else ()
            try:
                value = self.fget(*args)
            except Exception as exc:
                instance.errors[self.name] = exc
                value = self.default

        else:
            value = self.default

        setattr(instance, self.name, value)
        return value


class IndirectAttribute:
    def __init__(self, attr, source='_parent', missing=None):
        self.attr = attr
        self.source = source
        self.missing = missing

    def __get__(self, instance, cls=None):
        if instance is None:
            return self

        # Guarded getattr, returning None on error (and storing the
        # error or a missing message).
        source = getattr(instance, self.source)
        try:
            value = getattr(source, self.attr)
        except Exception as exc:
            instance.errors[self.attr] = exc
            value = None
        else:
            if value is None and self.missing:
                instance.missing[self.attr] = self.missing

        setattr(instance, self.attr, value)
        return value


class VLBIInfoMeta(type):
    # Ensure all attributes are initialized to None, so that they are
    # always available (do this rather than overwrite __getattr__ so that
    # we can generate docstrings in sphinx for them).
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        header0_attrs = dct.get('_header0_attrs', ())
        parent_attrs = dct.get('_parent_attrs', ())
        info_attrs = dct.get('info_names', ())
        not_set_attrs = tuple(attr for attr in (cls.attr_names
                                                + parent_attrs + info_attrs)
                              if not hasattr(cls, attr))
        for attr in not_set_attrs + header0_attrs:
            if attr in header0_attrs:
                setattr(cls, attr, IndirectAttribute(attr, source='header0'))
            elif attr in parent_attrs:
                setattr(cls, attr, IndirectAttribute(attr, source='_parent'))
            elif attr in info_attrs:
                setattr(cls, attr, info_property(OrderedDict, name=attr))


class VLBIInfoBase(metaclass=VLBIInfoMeta):
    """Container providing a standardized interface to file information.

    In order to ensure that information is always returned, all access
    to the parent should be within ``try/except`` with a possible error
    stored in ``self.errors``.  See ``self._getattr`` for an example.
    """

    attr_names = ()
    """Attributes that the container provides."""

    info_names = ()
    """Dictionaries with further information that the container provides."""

    _parent_attrs = ()
    _parent = None

    def _collect_info(self):
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
            info = instance.__dict__['info'] = self.__class__()
            info._parent = instance
            info._collect_info()

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
        """Create a dict with file information, including missing pieces."""
        info = {}
        if self:
            for attr in self.attr_names:
                value = getattr(self, attr)
                if value is not None:
                    info[attr] = value

        for info_name in self.info_names:
            extra = getattr(self, info_name)
            if extra:
                info[info_name] = extra

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
            if value is not None:
                if isinstance(value, Time):
                    value = Time(value, format='isot', precision=9)
                elif attr == 'sample_rate':
                    value = value.to(u.MHz)
                result += '{} = {}\n'.format(attr, value)

        for info_name in self.info_names:
            items = getattr(self, info_name, {})
            prefix = '\n{}: '.format(info_name)
            if info_name == 'missing':
                for msg in sorted(set(self.missing.values())):
                    keys = sorted(set(key for key in self.missing
                                      if self.missing[key] == msg))
                    result += "{} {}: {}\n".format(prefix,
                                                   ', '.join(keys), msg)
                    prefix = ' ' * (len(info_name) + 2)

            else:
                for key, value in items.items():
                    result += "{} {}: {}\n".format(prefix, key, str(value))
                    prefix = ' ' * (len(info_name) + 2)

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
                  'start_time', 'readable')
    _header0_attrs = ('bps', 'complex_data', 'samples_per_frame',
                      'sample_shape')
    info_names = ('missing', 'checks', 'errors', 'warnings')
    """Dictionaries with further information that the container provides."""

    @info_property
    def header0(self):
        # Here, we do not even know whether the file is open or whether we
        # have the right format. We thus use a try/except and filter out all
        # warnings.
        with self._parent.temporary_offset() as fh:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fh.seek(0)
                return fh.read_header()

    @info_property(needs='header0')
    def frame0(self):
        # Try reading a frame.  This has no business failing if a
        # frame rate could be determined, but try anyway; maybe file is closed.
        with self._parent.temporary_offset() as fh:
            fh.seek(0)
            return fh.read_frame()

    @info_property(needs='frame0', default=False)
    def decodable(self):
        # Getting the first sample can fail if we don't have the right decoder.
        first_sample = self.frame0[0]

        if not isinstance(first_sample, np.ndarray):
            raise TypeError('first sample is not an ndarray')

        return True

    @info_property(needs='header0')
    def format(self):
        return self._parent.__class__.__name__.split('File')[0].lower()

    @info_property(needs='header0')
    def frame_rate(self):
        return self._parent.get_frame_rate()

    @info_property(needs='header0')
    def number_of_frames(self):
        with self._parent.temporary_offset() as fh:
            number_of_frames = fh.seek(0, 2) / self.header0.frame_nbytes

        if number_of_frames % 1 == 0:
            return int(number_of_frames)
        else:
            self.warnings['number_of_frames'] = (
                'file contains non-integer number ({}) of frames'
                .format(number_of_frames))
            return None

    @info_property(needs='header0')
    def start_time(self):
        return self.header0.time

    @info_property(needs='frame0', default=False)
    def readable(self):
        self.checks['decodable'] = self.decodable
        return all(bool(v) for v in self.checks.values())

    @info_property(needs=('frame_rate', 'samples_per_frame'))
    def sample_rate(self):
        return self.frame_rate * self.samples_per_frame

    def __repr__(self):
        result = 'File information:\n'
        result += super().__repr__()
        return result


class VLBIStreamReaderInfo(VLBIInfoBase):
    """Standardized information on stream readers.

    The ``info`` descriptor provides a few standard attributes, all of which
    can also be accessed directly on the stream filehandle. More detailed
    information on the underlying file is stored in its info, accessible via
    ``info.file_info``.

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
                  'format', 'bps', 'complex_data', 'readable')
    """Attributes that the container provides."""

    _parent_attrs = tuple(attr for attr in attr_names
                          if attr not in ('format', 'readable'))

    info_names = ('checks', 'errors', 'warnings')
    """Dictionaries with further information that the container provides."""
    # Note that we cannot have missing information in streams;
    # they cannot be opened without it.

    @info_property
    def file_info(self):
        return self._parent.fh_raw.info

    @info_property(needs='file_info')
    def format(self):
        return self.file_info.format

    @info_property
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

    @info_property
    def readable(self):
        if self.file_info.readable:
            self.checks['continuous'] = self.continuous
            return all(bool(v) for v in self.checks.values())
        else:
            return False

    def _collect_info(self):
        super()._collect_info()
        # We want to include the raw info and its possible problems.
        for piece in self.info_names:
            file_piece = getattr(self.file_info, piece, None)
            if file_piece:
                stream_piece = file_piece.copy()
                stream_piece.update(getattr(self, piece))
                setattr(self, piece, stream_piece)

    def _up_to_date(self):
        # Stream readers cannot change after initialization, so check is easy.
        return True

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
            raw_info = file_info.info_names
            try:
                file_info.attr_names = [attr for attr in raw_attrs
                                        if attr not in self.attr_names]
                file_info.info_names = [name for name in raw_info
                                        if name not in self.info_names]
                result += '\n' + repr(file_info)
            finally:
                file_info.attr_names = raw_attrs
                file_info.info_names = raw_info

        return result
