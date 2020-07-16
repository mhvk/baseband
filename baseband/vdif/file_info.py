# Licensed under the GPLv3 - see LICENSE
"""The VDIFFileReaderInfo property.

Includes information about threads and frame sets.
"""
from ..base.file_info import FileReaderInfo, info_item


__all__ = ['VDIFFileReaderInfo']


class VDIFFileReaderInfo(FileReaderInfo):
    attr_names = (('format', 'edv', 'number_of_frames', 'thread_ids',
                   'number_of_framesets')
                  + FileReaderInfo.attr_names[2:])
    """Attributes that the container provides."""

    edv = info_item(needs='header0', doc=(
        'Extended data version (False if legacy).'))

    @info_item
    def thread_ids(self):
        # To get the thread_ids and thus the real sample shape,
        # need to check frame sets.
        with self._parent.temporary_offset(0) as fh:
            return fh.get_thread_ids()

    @info_item
    def header0(self):
        with self._parent.temporary_offset(0) as fh:
            # Almost all bytes are interpretable as headers,
            # so we need a basic sanity check.
            return fh.find_header(maximum=0)

    # Some headers also need frame rate, but fine to let that
    # lead to an error.
    @info_item(needs='header0')
    def start_time(self):
        return self.header0.get_time(frame_rate=self.frame_rate)

    @info_item(needs=('header0', 'thread_ids'))
    def sample_shape(self):
        return (len(self.thread_ids), self.header0.nchan)

    @info_item(needs=('number_of_frames', 'thread_ids'))
    def number_of_framesets(self):
        number_of_framesets = self.number_of_frames / len(self.thread_ids)
        if number_of_framesets % 1 == 0:
            return int(number_of_framesets)

        else:
            self.warnings['number_of_framesets'] = (
                'file contains non-integer number ({}) of '
                'framesets'.format(number_of_framesets))
            return None
