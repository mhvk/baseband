# Licensed under the GPLv3 - see LICENSE
from ..vlbi_base.file_info import (VLBIFileReaderInfo, VLBIStreamReaderInfo,
                                   info_item)


class GSBTimeStampInfo(VLBIFileReaderInfo):
    attr_names = ('format', 'mode', 'frame_rate', 'start_time', 'readable',
                  'missing', 'errors')
    _header0_attrs = ('mode',)
    # Should add number_of_frames, but tricky without _last_header.

    @info_item
    def header0(self):
        with self._parent.temporary_offset() as fh:
            fh.seek(0)
            return fh.read_timestamp()

    @info_item(needs='header0')
    def format(self):
        return 'gsb'

    # Cannot know whether it is readable without the raw data files.
    readable = None

    @info_item
    def missing(self):
        missing = super().missing
        missing['raw'] = 'need raw binary files for the stream reader'
        return missing


class GSBStreamReaderInfo(VLBIStreamReaderInfo):

    @info_item
    def frame0(self):
        return self._parent._read_frame(0)

    # Bit of a hack, but the base reader one suffices here with
    # the frame0 override above and its default "decodable"
    readable = VLBIFileReaderInfo.readable
    decodable = VLBIFileReaderInfo.decodable

    @info_item
    def file_info(self):
        return self._parent.fh_ts.info
