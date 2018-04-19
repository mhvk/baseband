# Licensed under the GPLv3 - see LICENSE
"""Provide a base class for "info" properties.

Loosely based on `~astropy.utils.data_info.DataInfo`.
"""
from __future__ import division, unicode_literals, print_function

import astropy.units as u
from astropy.utils.compat.misc import override__dir__


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

    # The standard attributes should always be accessible even if not defined,
    # so adjust attribute getting and dir'ing accordingly.
    def __getattr__(self, attr):
        if attr in self.attr_names:
            return None

        return self.__getattribute__(attr)

    @override__dir__
    def __dir__(self):
        return self.attr_names

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
            return 'File not parsable. Wrong format?'

        result = ''
        for attr in self.attr_names:
            value = getattr(self, attr)
            if value is not None:
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
        Entries in the dict are keyed by names of arguments that should be
        passed to the file reader to obtain full information. The associated
        entries in the dict explain why these arguments are needed.

    Examples
    --------
    The most common use is simply to print information::

        >>> from baseband.data import SAMPLE_MARK5B
        >>> from baseband import mark5b
        >>> fh = mark5b.open(SAMPLE_MARK5B, 'rb')
        >>> fh.info
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
        format = mark5b
        frame_rate = 6400.0 Hz
        sample_rate = 32.0 MHz
        samples_per_frame = 5000
        sample_shape = (8,)
        bps = 2
        complex_data = False
        start_time = 56821.22917824074
        >>> fh.close()
    """
    attr_names = ('format', 'frame_rate', 'sample_rate', 'samples_per_frame',
                  'sample_shape', 'bps', 'complex_data', 'start_time')
    _header0_attrs = ('bps', 'complex_data', 'samples_per_frame',
                      'sample_shape')

    def _get_header0(self):
        fh = self._parent
        old_offset = fh.tell()
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
            self.format = self._get_format()
            self.frame_rate = self._get_frame_rate()
            if (self.frame_rate is not None and
                    self.samples_per_frame is not None):
                self.sample_rate = (self.frame_rate *
                                    self.samples_per_frame).to(u.MHz)
            self.start_time = self._get_start_time()

    def __getattr__(self, attr):
        if not attr.startswith('_') and (self.header0 is not None and
                                         attr in self._header0_attrs):
            return getattr(self.header0, attr)

        return super(VLBIFileReaderInfo, self).__getattr__(attr)


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
    attr_names = ('format', 'start_time', 'stop_time', 'sample_rate',
                  'shape', 'bps', 'complex_data')
    _parent_attrs = attr_names[1:]

    def _raw_file_info(self):
        # mostly here so GSB can override.
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
        info['raw_file_info'] = self.file_info()
        return info
