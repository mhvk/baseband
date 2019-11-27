# Licensed under the GPLv3 - see LICENSE
"""Provide a base class for "info" properties.

Loosely based on `~astropy.utils.data_info.DataInfo`.
"""
import warnings
from collections import OrderedDict

import numpy as np
import astropy.units as u


__all__ = ['VLBIInfoMeta', 'VLBIInfoBase',
           'VLBIFileReaderInfo', 'VLBIStreamReaderInfo']


class VLBIInfoMeta(type):
    # Ensure all attributes are initialized to None, so that they are
    # always available (do this rather than overwrite __getattr__ so that
    # we can generate docstrings in sphinx for them).
    def __init__(cls, name, bases, dct):
        super().__init__(name, bases, dct)
        attr_names = dct.get('attr_names', ())
        for attr in attr_names:
            setattr(cls, attr, None)


class VLBIInfoBase(metaclass=VLBIInfoMeta):
    """Container providing a standardized interface to file information.

    In order to ensure that information is always returned, all access
    to the parent should be within ``try/except`` with a possible error
    stored in ``self.errors``.  See ``self._getattr`` for an example.
    """

    attr_names = ('format',)
    """Attributes that the container provides."""

    _parent_attrs = ()
    _parent = None

    def _getattr(self, object_, attr, error=True):
        """Guarded getattr, returning None on error (and storing the error)."""
        try:
            return getattr(object_, attr)
        except Exception as exc:
            if error:
                self.errors[attr] = exc
            return None

    def _collect_info(self):
        # We link to attributes from the parent rather than just overriding
        # __getattr__ to allow us to look for changes.
        self.missing = {}
        self.errors = OrderedDict()
        for attr in self._parent_attrs:
            setattr(self, attr, self._getattr(self._parent, attr))

    def _up_to_date(self):
        """Determine whether the information we have stored is up to date."""
        return all(getattr(self, attr) == self._getattr(self._parent, attr,
                                                        error=False)
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

    def __set__(self, info):
        # We do need to define __set__ since this ensures we are treated as
        # a "data descriptor", i.e., that our __get__ will get called even
        # if "info" is present in instance.__dict__; see
        # https://docs.python.org/3/howto/descriptor.html
        raise AttributeError("can't set info attribute.")

    def __bool__(self):
        return self.format is not None

    # PY2
    __nonzero__ = __bool__

    def __call__(self):
        """Create a dict with file information, including missing pieces."""
        info = {}
        if self:
            for attr in self.attr_names:
                value = getattr(self, attr)
                if value is not None:
                    info[attr] = value
            if self.missing:
                info['missing'] = self.missing

        if self.errors:
            info['errors'] = self.errors

        return info

    def __repr__(self):
        # Use the repr for quick display of file information.
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
                if hasattr(value, 'isot'):
                    value.precision = 9
                    value = value.isot
                elif attr == 'sample_rate':
                    value = value.to(u.MHz)
                result += '{} = {}\n'.format(attr, value)

        if self.missing:
            result += '\n'
            prefix = 'missing: '
            for msg in sorted(set(self.missing.values())):
                keys = sorted(set(key for key in self.missing
                                  if self.missing[key] == msg))
                result += "{} {}: {}\n".format(prefix, ', '.join(keys), msg)
                prefix = ' ' * len(prefix)

        if self.errors:
            result += '\n'
            prefix = 'errors: '
            for item, error in self.errors.items():
                result += "{} {}: {}\n".format(prefix, item, str(error))
                prefix = ' ' * len(prefix)

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
    errors : dict
        Any exceptions raised while trying to determine attributes.  Keyed
        by the attributes.

    Examples
    --------
    The most common use is simply to print information::

        >>> from baseband.data import SAMPLE_MARK5B
        >>> from baseband import mark5b
        >>> fh = mark5b.open(SAMPLE_MARK5B, 'rb')
        >>> fh.info
        File information:
        format = mark5b
        frame_rate = 6400.0 Hz
        bps = 2
        complex_data = False
        readable = False
        <BLANKLINE>
        missing:  nchan: needed to determine sample shape and rate.
                  kday, ref_time: needed to infer full times.
        <BLANKLINE>
        errors:  start_time: unsupported operand type(s) for +: ...
                 frame0: In order to read frames, the file handle ...

        >>> fh.close()

        >>> fh = mark5b.open(SAMPLE_MARK5B, 'rb', kday=56000, nchan=8)
        >>> fh.info
        File information:
        format = mark5b
        frame_rate = 6400.0 Hz
        sample_rate = 32.0 MHz
        samples_per_frame = 5000
        sample_shape = (8,)
        bps = 2
        complex_data = False
        start_time = 2014-06-13T05:30:01.000000000
        readable = True
        >>> fh.close()
    """
    attr_names = ('format', 'frame_rate', 'sample_rate', 'samples_per_frame',
                  'sample_shape', 'bps', 'complex_data', 'start_time',
                  'readable')
    _header0_attrs = ('bps', 'complex_data', 'samples_per_frame',
                      'sample_shape')

    def _get_header0(self):
        # Here, we do not even know whether the file is open or whether we
        # have the right format. We thus use a try/except and filter out all
        # warnings.
        try:
            with self._parent.temporary_offset() as fh:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    fh.seek(0)
                    return fh.read_header()
        except Exception as exc:
            self.errors['header0'] = exc
            return None

    def _get_frame0(self):
        # Try reading a frame.  This has no business failing if a
        # frame rate could be determined, but try anyway; maybe file is closed.
        try:
            with self._parent.temporary_offset() as fh:
                fh.seek(0)
                return fh.read_frame()
        except Exception as exc:
            self.errors['frame0'] = exc
            return None

    def _readable(self):
        frame0 = self._get_frame0()
        if frame0 is None:
            return False

        # Getting the first sample can fail if we don't have the right decoder.
        try:
            first_sample = frame0[0]
        except Exception as exc:
            self.errors['readable'] = exc
            return False

        if not isinstance(first_sample, np.ndarray):
            self.errors['readable'] = 'first sample is not an ndarray'
            return False

        return True

    def _get_format(self):
        return self._parent.__class__.__name__.split('File')[0].lower()

    def _get_frame_rate(self):
        try:
            return self._parent.get_frame_rate()
        except Exception as exc:
            self.errors['frame_rate'] = exc
            return None

    def _get_start_time(self):
        try:
            return self.header0.time
        except Exception as exc:
            self.errors['start_time'] = exc
            return None

    def _collect_info(self):
        super()._collect_info()
        self.header0 = self._get_header0()
        if self.header0 is not None:
            for attr in self._header0_attrs:
                setattr(self, attr, getattr(self.header0, attr))
            self.format = self._get_format()
            self.frame_rate = self._get_frame_rate()
            if ('sample_rate' not in self._header0_attrs
                    and self.frame_rate is not None
                    and self.samples_per_frame is not None):
                self.sample_rate = self.frame_rate * self.samples_per_frame
            self.start_time = self._get_start_time()
            self.readable = self._readable()

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
        Whether the first sample could be read and decoded.
    """
    attr_names = ('start_time', 'stop_time', 'sample_rate', 'shape',
                  'format', 'bps', 'complex_data', 'readable')
    _parent_attrs = tuple(attr for attr in attr_names
                          if attr not in ('format', 'readable'))

    def _raw_file_info(self):
        # Mostly here so GSB can override.
        return self._parent.fh_raw.info

    def _readable(self):
        # Again mostly here so GSB can override.
        return self._parent.readable()

    def _collect_info(self):
        super()._collect_info()
        # We also want the raw info.
        self.file_info = self._raw_file_info()
        self.format = self.file_info.format
        self.readable = self._readable()

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
            # Add information from the raw file.
            raw_attrs = file_info.attr_names
            raw_only_attrs = [attr for attr in raw_attrs
                              if attr not in self.attr_names]
            try:
                file_info.attr_names = raw_only_attrs
                result += '\n' + repr(file_info)
            finally:
                file_info.attr_names = raw_attrs

        return result
