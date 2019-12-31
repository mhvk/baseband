# Licensed under the GPLv3 - see LICENSE
from ..vlbi_base.file_info import VLBIFileReaderInfo, info_property


class VDIFFileReaderInfo(VLBIFileReaderInfo):
    attr_names = (('format', 'edv', 'number_of_frames', 'thread_ids',
                   'number_of_framesets')
                  + VLBIFileReaderInfo.attr_names[2:])
    _header0_attrs = ('edv', 'bps', 'samples_per_frame')

    def _get_thread_ids(self):
        # To get the thread_ids and thus the real sample shape,
        # need to check frame sets.
        try:
            with self._parent.temporary_offset() as fh:
                fh.seek(0)
                return fh.get_thread_ids()
        except Exception as exc:
            self.errors['thread_ids'] = exc
            return None

    @info_property
    def header0(self):
        with self._parent.temporary_offset() as fh:
            fh.seek(0)
            # Almost all bytes are interpretable as headers,
            # so we need a basic sanity check.
            return fh.find_header(maximum=0)

    @info_property
    def start_time(self):
        return self.header0.get_time(frame_rate=self.frame_rate)

    def _collect_info(self):
        super()._collect_info()
        if self:
            self.complex_data = self.header0['complex_data']
            self.thread_ids = self._get_thread_ids()
            if self.thread_ids is not None:
                n_threads = len(self.thread_ids)
                self.sample_shape = (n_threads, self.header0.nchan)
                if self.number_of_frames is not None:
                    number_of_framesets = self.number_of_frames / n_threads
                    if number_of_framesets % 1 == 0:
                        self.number_of_framesets = int(number_of_framesets)
                    else:
                        self.warnings['number_of_framesets'] = (
                            'file contains non-integer number ({}) of '
                            'framesets'.format(number_of_framesets))
