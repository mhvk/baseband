# Licensed under the GPLv3 - see LICENSE
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ..vlbi_base.file_info import VLBIFileReaderInfo


class Mark4FileReaderInfo(VLBIFileReaderInfo):
    attr_names = VLBIFileReaderInfo.attr_names + ('ntrack', 'offset0')
    _header0_attrs = ('bps', 'samples_per_frame')
    _parent_attrs = ('ntrack', 'decade', 'ref_time')

    def _get_header0(self):
        fh = self._parent
        old_offset = fh.tell()
        try:
            fh.seek(0)
            offset0 = fh.locate_frame()
            if offset0 is None:
                return None

            self.offset0 = offset0
            return fh.read_header()
        except Exception:
            return None
        finally:
            fh.seek(old_offset)

    def _collect_info(self):
        super(Mark4FileReaderInfo, self)._collect_info()
        if self:
            self.complex_data = False
            # TODO: Shouldn't Mark4Header provide this?
            self.sample_shape = (self.header0.nchan,)
            if self.decade is None and self.ref_time is None:
                self.missing['decade'] = self.missing['ref_time'] = (
                    "needed to infer full times.")
