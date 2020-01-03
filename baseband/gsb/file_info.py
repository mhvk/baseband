# Licensed under the GPLv3 - see LICENSE
from ..vlbi_base.file_info import (VLBIFileReaderInfo, VLBIStreamReaderInfo,
                                   info_item)


class GSBTimeStampInfo(VLBIFileReaderInfo):
    attr_names = ('format', 'mode', 'number_of_frames', 'frame_rate',
                  'start_time', 'readable', 'missing', 'errors', 'warnings')
    _header0_attrs = ('mode',)

    @info_item
    def header0(self):
        with self._parent.temporary_offset() as fh:
            fh.seek(0)
            return fh.read_timestamp()

    @info_item(needs='header0')
    def format(self):
        return 'gsb'

    @info_item(needs='header0')
    def number_of_frames(self):
        with self._parent.temporary_offset() as fh:
            file_nbytes = fh.seek(0, 2)
            if file_nbytes == self.header0.nbytes:
                return 1

            guess = max(int(round(file_nbytes / self.header0.nbytes)) - 2, 0)
            while True:
                guess += 1
                fh.seek(self.header0.seek_offset(guess))
                try:
                    self.header0.fromfile(fh).time
                except EOFError:
                    break

        return guess

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
