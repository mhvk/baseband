# Licensed under the GPLv3 - see LICENSE
from ..vlbi_base.file_info import VLBIFileReaderInfo


class GUPPIFileReaderInfo(VLBIFileReaderInfo):
    # Get sample_rate from header rather than calculate it from frame_rate
    # and samples_per_frame, since we need to correct for overlap.
    _header0_attrs = VLBIFileReaderInfo._header0_attrs + ('sample_rate',)
