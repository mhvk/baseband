# Licensed under the GPLv3 - see LICENSE
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ..vlbi_base.file_info import VLBIFileReaderInfo, VLBIInfoBase


class GSBTimeStampInfo(VLBIFileReaderInfo):
    attr_names = ('format', 'mode') + VLBIFileReaderInfo.attr_names[1:]
    _header0_attrs = ('mode',)

    def _get_header0(self):
        fh = self._parent
        old_offset = fh.tell()
        try:
            fh.seek(0)
            return fh.read_timestamp()
        except Exception:
            return None
        finally:
            fh.seek(old_offset)

    def _get_format(self):
        return 'gsb'

    def _collect_info(self):
        super(GSBTimeStampInfo, self)._collect_info()
        if self:
            self.missing['raw'] = 'need raw binary files for the stream reader'


class GSBStreamReaderInfo(VLBIInfoBase):
    _parent_attrs = ('samples_per_frame', 'bps', 'complex_data',
                     'sample_shape', 'sample_rate', 'stop_time', 'size')

    def _collect_info(self):
        super(GSBStreamReaderInfo, self)._collect_info()
        # Part of our information, including the format, comes from the
        # underlying timestamp file.
        self._fh_ts_info = self._parent.fh_ts.info
        self._fh_ts_info_attrs = self._fh_ts_info.attr_names
        extra_attrs = tuple(attr for attr in self._parent_attrs
                            if attr not in self._fh_ts_info_attrs)
        self.attr_names = self._fh_ts_info_attrs + extra_attrs

    def _up_to_date(self):
        # Stream readers cannot after initialization, so the check is easy.
        return True

    def __getattr__(self, attr):
        if not attr.startswith('_') and attr in self._fh_ts_info_attrs:
            return getattr(self._fh_ts_info, attr)

        return super(GSBStreamReaderInfo, self).__getattr__(attr)
