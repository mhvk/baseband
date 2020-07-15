# Licensed under the GPLv3 - see LICENSE
"""The Mark5BFileReaderInfo property.

Includes information about what is needed to calcuate times.
"""
from ..base.file_info import FileReaderInfo, info_item


__all__ = ['Mark5BFileReaderInfo']


class Mark5BFileReaderInfo(FileReaderInfo):
    _header0_attrs = ('complex_data',)
    _parent_attrs = ('nchan', 'bps', 'ref_time', 'kday')

    bps = info_item(needs='_parent', missing='needed to decode data')
    nchan = info_item(needs='_parent', missing=(
        "needed to determine sample shape, frame rate, decode data."))

    @info_item
    def time_info(self):
        """Additional time info needed to get the start time."""
        time_info = (self.kday, self.ref_time)
        if time_info == (None, None):
            self.missing['kday'] = self.missing['ref_time'] = (
                "needed to infer full times.")
            return None

        return time_info

    @info_item(needs=('header0', 'frame_rate', 'time_info'))
    def start_time(self):
        """Time of the first sample."""
        return self.header0.get_time(frame_rate=self.frame_rate)

    @info_item(needs='nchan')
    def sample_shape(self):
        """Dimensions of each complete sample."""
        return (self.nchan,)

    @info_item(needs=('header0', 'bps', 'nchan'))
    def samples_per_frame(self):
        """Number of complete samples in each frame."""
        return (self.header0.payload_nbytes * 8
                // (self.bps * self.nchan))

    # Override just to replace what it "needs".
    @info_item
    def format(self):
        with self._parent.temporary_offset(0):
            return 'mark5b' if self._parent.locate_frames() else None

    @info_item(needs=('header0', 'bps', 'nchan'))
    def frame0(self):
        """First frame from the file."""
        return super().frame0
