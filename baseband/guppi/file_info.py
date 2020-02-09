# Licensed under the GPLv3 - see LICENSE
"""The GuppiFileReaderInfo property.

Overrides what can be gotten from the first header.
"""
from ..vlbi_base.file_info import VLBIFileReaderInfo


__all__ = ['GUPPIFileReaderInfo']


class GUPPIFileReaderInfo(VLBIFileReaderInfo):
    # Get sample_rate from header rather than calculate it from frame_rate
    # and samples_per_frame, since we need to correct for overlap.
    _header0_attrs = VLBIFileReaderInfo._header0_attrs + ('sample_rate',)
