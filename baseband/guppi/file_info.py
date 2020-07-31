# Licensed under the GPLv3 - see LICENSE
"""The GuppiFileReaderInfo property.

Overrides what can be gotten from the first header.
"""
from ..base.file_info import FileReaderInfo, info_item


__all__ = ['GUPPIFileReaderInfo']


class GUPPIFileReaderInfo(FileReaderInfo):
    # Get sample_rate from header rather than calculate it from frame_rate
    # and samples_per_frame, since we need to correct for overlap.
    attr_names = list(FileReaderInfo.attr_names)
    attr_names.insert(attr_names.index('format')+1, 'pktfmt')
    attr_names.insert(attr_names.index('samples_per_frame')+1, 'overlap')
    attr_names = tuple(attr_names)
    """Attributes that the container provides."""

    overlap = info_item(needs='header0', doc=(
        'Number of complete samples that overlap between frames.'))
    sample_rate = info_item(needs='header0', doc=(
        'Number of complete samples per second.'))

    @info_item(needs='header0')
    def pktfmt(self):
        """Packet format for the data."""
        pktfmt = self.header0['PKTFMT']
        if pktfmt not in self.header0.supported_formats:
            self.warnings['pktfmt'] = (f'Unknown pktfmt {pktfmt!r}. '
                                       f'Assuming channels are stored first.')
        return pktfmt
