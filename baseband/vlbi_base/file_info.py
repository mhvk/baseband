# Licensed under the GPLv3 - see LICENSE
"""Provide a base class for "info" properties.

Loosely based on astropy.utils.data_info.DataInfo
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
        # Check if we have a stored and up to date copy.
        info = instance.__dict__.get('info')
        if info is None or not info._up_to_date():
            # If not, create a new instance and fill it.
            # Note: cannot change "self", as this was created on the class.
            info = instance.__dict__['info'] = self.__class__()
            info._parent = instance
            info._collect_info()

        return info

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
            for msg in set(self.missing.values()):
                keys = set(key for key in self.missing
                           if self.missing[key] == msg)
                result += "{} {}: {}\n".format(prefix, ', '.join(keys), msg)
                prefix = ' ' * len(prefix)

        return result


class VLBIFileReaderInfo(VLBIInfoBase):
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

    def _get_start_time(self):
        try:
            return self.header0.time
        except Exception:
            return None

    def _get_frame_rate(self):
        try:
            return self._parent.get_frame_rate()
        except Exception:
            return None

    def _collect_info(self):
        super(VLBIFileReaderInfo, self)._collect_info()
        self.header0 = self._get_header0()
        if self.header0 is not None:
            self.format = self._get_format()
            self.start_time = self._get_start_time()
            self.frame_rate = self._get_frame_rate()
            if (self.frame_rate is not None and
                    self.samples_per_frame is not None):
                self.sample_rate = (self.frame_rate *
                                    self.samples_per_frame).to(u.MHz)

    def __getattr__(self, attr):
        if not attr.startswith('_') and (self.header0 is not None and
                                         attr in self._header0_attrs):
            return getattr(self.header0, attr)

        return super(VLBIFileReaderInfo, self).__getattr__(attr)


class VLBIStreamReaderInfo(VLBIInfoBase):
    _parent_attrs = ('sample_shape', 'sample_rate', 'stop_time', 'size')

    def _collect_info(self):
        super(VLBIStreamReaderInfo, self)._collect_info()
        # Part of our information, including the format, comes from the
        # underlying raw file.
        self._fh_raw_info = self._parent.fh_raw.info
        self._fh_raw_info_attrs = self._fh_raw_info.attr_names
        extra_attrs = tuple(attr for attr in self._parent_attrs
                            if attr not in self._fh_raw_info_attrs)
        self.attr_names = self._fh_raw_info_attrs + extra_attrs

    def _up_to_date(self):
        # Stream readers cannot after initialization, so the check is easy.
        return True

    def __getattr__(self, attr):
        if not attr.startswith('_') and attr in self._fh_raw_info_attrs:
            return getattr(self._fh_raw_info, attr)

        return super(VLBIStreamReaderInfo, self).__getattr__(attr)
