from ..base.base import StreamReaderBase
from .frame import ASPFrame
from .header import ASPFileHeader, ASPHeader
import astropy.units as u


__all__ = ['ASPStreamReaderBase', 'ASPStreamReader']


class ASPStreamReaderBase(StreamReaderBase):

    # unfinished
    def __init__(self, fh_raw, header0):
        self._fh_raw = fh_raw
        self._header0 = header0
        sample_rate = header0['ch_bw'][0] * u.MHz
        samples_per_frame = header0['NPtsSend'][0]
        super().__init__(
            fh_raw, header0, sample_rate=sample_rate,
            samples_per_frame=samples_per_frame,
            sample_shape=None, bps=8, complex_data=True,
            squeeze=False, subset=None, fill_value=0.0, verify=False)

    def read_frame(self):
        frame = ASPFrame.fromfile(self._fh_raw)
        # associate the original file header with
        # frame header (just in case!)
        frame.header.file_header = self._header0.file_header
        return frame


class ASPStreamReader(ASPStreamReaderBase):
    def __init__(self, fh_raw):
        pos = fh_raw.tell()
        fh_raw.seek(0)
        fileheader0 = ASPFileHeader.fromfile(fh_raw)
        pos2 = fh_raw.tell()
        header0 = ASPHeader.fromfile(fh_raw)
        header0.file_header = fileheader0

        if pos == 0:
            pos = pos2

        fh_raw.seek(pos)

        super().__init__(fh_raw, header0)
