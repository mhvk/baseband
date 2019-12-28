# Licensed under the GPLv3 - see LICENSE
from ..vlbi_base.file_info import VLBIFileReaderInfo


class VDIFFileReaderInfo(VLBIFileReaderInfo):
    attr_names = (('format', 'edv', 'thread_ids')
                  + VLBIFileReaderInfo.attr_names[1:])
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

    def _get_header0(self):
        try:
            with self._parent.temporary_offset() as fh:
                fh.seek(0)
                header0 = fh.read_header()
                # Almost all bytes are interpretable as headers,
                # so we need a basic sanity check.
                fh.seek(header0.frame_nbytes)
                header1 = fh.read_header()
                if header1.same_stream(header0):
                    return header0
        except Exception as exc:
            self.errors['header0'] = exc
            return None

    def _get_start_time(self):
        try:
            return self.header0.get_time(frame_rate=self.frame_rate)
        except Exception as exc:
            self.errors['start_time'] = exc
            return None

    def _collect_info(self):
        super()._collect_info()
        if self:
            self.complex_data = self.header0['complex_data']
            self.thread_ids = self._get_thread_ids()
            if self.thread_ids is not None:
                self.sample_shape = (len(self.thread_ids), self.header0.nchan)
