# Licensed under the GPLv3 - see LICENSE
from ..vlbi_base.file_info import (VLBIFileReaderInfo, VLBIStreamReaderInfo,
                                   info_property)


class GSBTimeStampInfo(VLBIFileReaderInfo):
    attr_names = ('format', 'mode') + VLBIFileReaderInfo.attr_names[1:]
    _header0_attrs = ('mode',)

    @info_property
    def header0(self):
        with self._parent.temporary_offset() as fh:
            fh.seek(0)
            return fh.read_timestamp()

    @info_property
    def format(self):
        return 'gsb' if self.header0 is not None else None

    # Cannot know whether it is readable without the raw data files.
    readable = None

    number_of_frames = None
    # Tricky to determine without _last_header.

    def _collect_info(self):
        super()._collect_info()
        if self:
            self.missing['raw'] = 'need raw binary files for the stream reader'


class GSBStreamReaderInfo(VLBIStreamReaderInfo):

    @info_property
    def frame0(self):
        return self._parent._read_frame(0)

    # Bit of a hack, but the base reader one suffices here with
    # the frame0 override above and its default "decodable"
    readable = VLBIFileReaderInfo.readable
    decodable = VLBIFileReaderInfo.decodable

    @info_property
    def file_info(self):
        return self._parent.fh_ts.info
