from ..base.base import StreamReaderBase, FileOpener
from .frame import ASPFrame
from .header import ASPHeader, ASPFileHeader


__all__ = ['ASPStreamReaderBase', 'ASPStreamReader']


class ASPStreamReaderBase(StreamReaderBase):

    # unfinished
    def __init__(self, fh_raw, header0):
        super().__init__(
            fh_raw, header0, bps=8, complex_data=True)

    def read_frame(self):
        frame = ASPFrame.fromfile(self._fh_raw)
        # associate the original file header with
        # frame header (just in case!)
        frame.header.file_header = self._header0.file_header
        return frame


class ASPStreamReader(ASPStreamReaderBase):

    def __init__(self, fh_raw):
        file_header = ASPFileHeader.fromfile(fh_raw)
        header0 = ASPHeader.fromfile(fh_raw, file_header=file_header)
        super().__init__(fh_raw, header0)


open = FileOpener('ASP', header_class=ASPHeader, classes={
    'rs': ASPStreamReader}).wrapped(module=__name__, doc="")
