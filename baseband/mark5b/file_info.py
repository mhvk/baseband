# Licensed under the GPLv3 - see LICENSE
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ..vlbi_base.file_info import VLBIFileReaderInfo


class Mark5BFileReaderInfo(VLBIFileReaderInfo):
    _header0_attrs = ()
    _parent_attrs = ('nchan', 'bps', 'kday', 'ref_time')

    def _get_start_time(self):
        try:
            return self.header0.get_time(frame_rate=self.frame_rate)
        except Exception:
            return None

    def _collect_info(self):
        super(Mark5BFileReaderInfo, self)._collect_info()
        if self:
            self.complex_data = False
            if self.kday is None and self.ref_time is None:
                self.missing['kday'] = self.missing['ref_time'] = (
                    "needed to infer full times.")

            if self.nchan is None:
                self.missing['nchan'] = (
                    "needed to determine sample shape and rate.")
            else:
                self.sample_shape = (self.nchan,)
                self.samples_per_frame = (self.header0.payload_nbytes * 8 //
                                          (self.bps * self.nchan))
                self.sample_rate = self.samples_per_frame * self.frame_rate
