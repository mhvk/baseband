# Licensed under the GPLv3 - see LICENSE
from ..vlbi_base.file_info import VLBIFileReaderInfo, VLBIStreamReaderInfo


class GSBTimeStampInfo(VLBIFileReaderInfo):
    attr_names = ('format', 'mode') + VLBIFileReaderInfo.attr_names[1:]
    _header0_attrs = ('mode',)

    def _get_header0(self):
        try:
            with self._parent.temporary_offset() as fh:
                fh.seek(0)
                return fh.read_timestamp()
        except Exception as exc:
            self.errors['header0'] = exc
            return None

    def _get_format(self):
        return 'gsb'

    def _readable(self):
        # Cannot know whether it is readable without the raw data files.
        return None

    def _get_number_of_frames(self):
        # Tricky to determine without _last_header.
        return None

    def _collect_info(self):
        super()._collect_info()
        if self:
            self.missing['raw'] = 'need raw binary files for the stream reader'


class GSBStreamReaderInfo(VLBIStreamReaderInfo):

    def _get_frame0(self):
        try:
            return self._parent._read_frame(0)
        except Exception as exc:
            self.errors['frame0'] = exc
            return None

    _decodable = VLBIFileReaderInfo._decodable

    def _readable(self):
        # Bit of a hack, but the base reader one suffices here with
        # the _get_frame0 override above.
        return VLBIFileReaderInfo._readable(self)

    def _file_info(self):
        return self._parent.fh_ts.info
