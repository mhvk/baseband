# Licensed under the GPLv3 - see LICENSE
from ..vlbi_base.file_info import VLBIFileReaderInfo, info_property


class Mark5BFileReaderInfo(VLBIFileReaderInfo):
    _header0_attrs = ()
    _parent_attrs = ('nchan', 'bps', 'ref_time', 'kday')

    @info_property
    def time_info(self):
        time_info = (self.kday, self.ref_time)
        if time_info == (None, None):
            self.missing['kday'] = self.missing['ref_time'] = (
                "needed to infer full times.")
            return None

        return time_info

    @info_property
    def bps(self):
        bps = self._parent.bps
        if bps is None:
            self.missing['bps'] = 'needed to decode data'

        return bps

    @info_property
    def nchan(self):
        nchan = self._parent.nchan
        if nchan is None:
            self.missing['nchan'] = (
                "needed to determine sample shape, frame rate, decode data.")
        return nchan

    @info_property(needs=('header0', 'frame_rate', 'time_info'))
    def start_time(self):
        return self.header0.get_time(frame_rate=self.frame_rate)

    @info_property(needs='nchan')
    def sample_shape(self):
        return (self.nchan,)

    @info_property(needs=('header0', 'bps', 'nchan'))
    def samples_per_frame(self):
        return (self.header0.payload_nbytes * 8
                // (self.bps * self.nchan))

    @info_property(needs=('header0', 'bps', 'nchan'))
    def frame0(self):
        return super().frame0

    complex_data = False
