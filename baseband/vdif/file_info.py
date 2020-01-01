# Licensed under the GPLv3 - see LICENSE
from ..vlbi_base.file_info import VLBIFileReaderInfo, info_property


class VDIFFileReaderInfo(VLBIFileReaderInfo):
    attr_names = (('format', 'edv', 'number_of_frames', 'thread_ids',
                   'number_of_framesets')
                  + VLBIFileReaderInfo.attr_names[2:])
    _header0_attrs = ('edv', 'bps', 'samples_per_frame')

    @info_property
    def thread_ids(self):
        # To get the thread_ids and thus the real sample shape,
        # need to check frame sets.
        with self._parent.temporary_offset() as fh:
            fh.seek(0)
            return fh.get_thread_ids()

    @info_property
    def header0(self):
        with self._parent.temporary_offset() as fh:
            fh.seek(0)
            # Almost all bytes are interpretable as headers,
            # so we need a basic sanity check.
            return fh.find_header(maximum=0)

    @info_property(needs=('header0', 'frame_rate'))
    def start_time(self):
        return self.header0.get_time(frame_rate=self.frame_rate)

    @info_property(needs=('header0', 'thread_ids'))
    def sample_shape(self):
        return (len(self.thread_ids), self.header0.nchan)

    @info_property(needs=('number_of_frames', 'thread_ids'))
    def number_of_framesets(self):
        number_of_framesets = self.number_of_frames / len(self.thread_ids)
        if number_of_framesets % 1 == 0:
            return int(number_of_framesets)

        else:
            self.warnings['number_of_framesets'] = (
                'file contains non-integer number ({}) of '
                'framesets'.format(number_of_framesets))
            return None

    @info_property(needs='header0')
    def complex_data(self):
        return self.header0['complex_data']
