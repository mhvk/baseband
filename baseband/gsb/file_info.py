# Licensed under the GPLv3 - see LICENSE
from ..vlbi_base.file_info import VLBIFileReaderInfo, VLBIStreamReaderInfo


class GSBTimeStampInfo(VLBIFileReaderInfo):
    attr_names = ('format', 'mode') + VLBIFileReaderInfo.attr_names[1:]
    _header0_attrs = ('mode',)

    def _get_header0(self):
        with self._parent.temporary_offset() as fh:
            try:
                fh.seek(0)
                return fh.read_timestamp()
            except Exception as exc:
                self.errors['header0'] = exc
                return None

    def _get_format(self):
        return 'gsb'

    def _readable(self):
        return "unknown: need raw binary files"

    def _collect_info(self):
        super()._collect_info()
        if self:
            self.missing['raw'] = 'need raw binary files for the stream reader'


class GSBStreamReaderInfo(VLBIStreamReaderInfo):

    def _raw_file_info(self):
        info = self._parent.fh_ts.info
        # The timestamp reader info has a built-in missing for the
        # raw file, but this is incorrect if we're in a stream, which
        # cannot have been opened without one. (Yes, this is a hack.)
        info.missing = {}
        return info
