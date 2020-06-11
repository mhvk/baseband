# Licensed under the GPLv3 - see LICENSE
from ..vlbi_base.file_info import (VLBIFileReaderInfo, VLBIStreamReaderInfo,
                                   info_item)


class GSBTimeStampInfo(VLBIFileReaderInfo):
    attr_names = ('format', 'mode', 'number_of_frames', 'frame_rate',
                  'start_time', 'readable', 'missing', 'errors', 'warnings')
    _header0_attrs = ('mode',)

    @info_item
    def header0(self):
        with self._parent.temporary_offset(0) as fh:
            return fh.read_timestamp()

    @info_item(needs='header0')
    def format(self):
        return 'gsb'

    @info_item(needs='header0')
    def number_of_frames(self):
        with self._parent.temporary_offset() as fh:
            fh_size = fh.seek(0, 2)
            # Guess based on a fixed header size.  In reality, this
            # may be an overestimate as the headers can grow in size,
            # or an underestimate as the last header may be partial.
            # So, search around to be sure.
            guess = max(fh_size // self.header0.nbytes, 1)
            while self.header0.seek_offset(guess) > fh_size:
                guess -= 1
            while self.header0.seek_offset(guess) < fh_size:
                guess += 1

            # Now see if there is indeed a nice header before.
            fh.seek(self.header0.seek_offset(guess-1))
            line_tuple = fh.readline().split()
            # But realize that sometimes an incomplete header is written.
            if (len(" ".join(line_tuple))
                    < len(" ".join(self.header0.words))):
                self.warnings['number_of_frames'] = (
                    'last header is incomplete and is ignored')
                retry = True
            else:
                # Check last header is readable.
                try:
                    self.header0.__class__(line_tuple).time
                except Exception as exc:
                    self.warnings['number_of_frames'] = (
                        'last header failed to read ({}) and is ignored'
                        .format(str(exc)))
                    retry = True
                else:
                    retry = False
            if retry:
                guess -= 1
                fh.seek(self.header0.seek_offset(guess-1))
                self.header0.fromfile(fh).time

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
        fh_ts_info = self._parent.fh_ts.info
        fh_ts_info.missing.pop('raw', None)
        return fh_ts_info

    @info_item
    def payload_nbytes(self):
        return self._parent.payload_nbytes
