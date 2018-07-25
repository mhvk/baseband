# Licensed under the GPLv3 - see LICENSE
"""Provide a base class for "info" properties.

Loosely based on `~astropy.utils.data_info.DataInfo`.
"""
from __future__ import division, unicode_literals, print_function

import warnings

import astropy.units as u
from astropy.extern import six


class VLBIInfoMeta(type):
    # Ensure all attributes are initialized to None, so that they are
    # always available (do this rather than overwrite __getattr__ so that
    # we can generate docstrings in sphinx for them).
    def __init__(cls, name, bases, dct):
        super(VLBIInfoMeta, cls).__init__(name, bases, dct)
        attr_names = dct.get('attr_names', ())
        for attr in attr_names:
            setattr(cls, attr, None)


@six.add_metaclass(VLBIInfoMeta)
class VLBIInfoBase(object):
    """Container providing a standardized interface to file information."""

    attr_names = ('format',)
    """Attributes that the container provides."""

    _parent_attrs = ()
    _parent = None

    def _collect_info(self):
        # We link to attributes from the parent rather than just overriding
        # __getattr__ to allow us to look for changes.
        for attr in self._parent_attrs:
            setattr(self, attr, getattr(self._parent, attr))
        self.missing = {}

    def _up_to_date(self):
        """Determine whether the information we have stored is up to date."""
        return all(getattr(self, attr) == getattr(self._parent, attr)
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

        return info

    def __repr__(self):
        # Use the repr for quick display of file information.
        if self._parent is None:
            return super(VLBIInfoBase, self).__repr__()

        if not self:
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
    missing : dict
        Entries are keyed by names of arguments that should be passed to
        the file reader to obtain full information. The associated entries
        explain why these arguments are needed.

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
        <BLANKLINE>
        missing:  nchan: needed to determine sample shape and rate.
                  kday, ref_time: needed to infer full times.
        <BLANKLINE>
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
        >>> fh.close()
    """
    attr_names = ('format', 'frame_rate', 'sample_rate', 'samples_per_frame',
                  'sample_shape', 'bps', 'complex_data', 'start_time')
    _header0_attrs = ('bps', 'complex_data', 'samples_per_frame',
                      'sample_shape')

    def _get_header0(self):
        fh = self._parent
        old_offset = fh.tell()
        # Here, we do not even know whether we have the right format. We thus
        # use a try/except and filter out all warnings.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                fh.seek(0)
                return fh.read_header()
            except Exception:
                return None
            finally:
                fh.seek(old_offset)

    def _get_format(self):
        return self._parent.__class__.__name__.split('File')[0].lower()

    def _get_frame_rate(self):
        try:
            return self._parent.get_frame_rate()
        except Exception:
            return None

    def _get_start_time(self):
        try:
            return self.header0.time
        except Exception:
            return None

    def _collect_info(self):
        super(VLBIFileReaderInfo, self)._collect_info()
        self.header0 = self._get_header0()
        if self.header0 is not None:
            for attr in self._header0_attrs:
                setattr(self, attr, getattr(self.header0, attr))
            self.format = self._get_format()
            self.frame_rate = self._get_frame_rate()
            if (self.frame_rate is not None and
                    self.samples_per_frame is not None):
                self.sample_rate = self.frame_rate * self.samples_per_frame
            self.start_time = self._get_start_time()

    def __repr__(self):
        result = 'File information:\n'
        result += super(VLBIFileReaderInfo, self).__repr__()
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
    """
    attr_names = ('start_time', 'stop_time', 'sample_rate', 'shape',
                  'format', 'bps', 'complex_data')
    _parent_attrs = tuple(attr for attr in attr_names if attr != 'format')

    def _raw_file_info(self):
        # Mostly here so GSB can override.
        return self._parent.fh_raw.info

    def _collect_info(self):
        super(VLBIStreamReaderInfo, self)._collect_info()
        # We also want the raw info.
        self.file_info = self._raw_file_info()
        self.format = self.file_info.format

    def _up_to_date(self):
        # Stream readers cannot after initialization, so the check is easy.
        return True

    def __call__(self):
        """Create a dict with information about the stream and the raw file."""
        info = super(VLBIStreamReaderInfo, self).__call__()
        info['file_info'] = self.file_info()
        return info

    def __repr__(self):
        result = 'Stream information:\n'
        result += super(VLBIStreamReaderInfo, self).__repr__()
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
