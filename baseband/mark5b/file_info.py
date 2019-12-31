# Licensed under the GPLv3 - see LICENSE
from ..vlbi_base.file_info import VLBIFileReaderInfo, info_property


class Mark5BFileReaderInfo(VLBIFileReaderInfo):
    _header0_attrs = ()
    _parent_attrs = ('nchan', 'bps', 'kday', 'ref_time')

    @info_property(needs=('header0', 'frame_rate'))
    def start_time(self):
        return self.header0.get_time(frame_rate=self.frame_rate)

    @info_property(needs='nchan')
    def sample_shape(self):
        return (self.nchan,)

    @info_property(needs=('header0', 'bps', 'nchan'))
    def samples_per_frame(self):
        return (self.header0.payload_nbytes * 8
                // (self.bps * self.nchan))

    complex_data = False

    def _collect_info(self):
        super()._collect_info()
        if self:
            if self.kday is None and self.ref_time is None:
                self.missing['kday'] = self.missing['ref_time'] = (
                    "needed to infer full times.")

            if self.nchan is None:
                self.missing['nchan'] = (
                    "needed to determine sample shape and rate.")
