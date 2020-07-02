from astropy import units as u

from ..base.base import FileBase, StreamReaderBase, FileOpener
from ..base.file_info import FileReaderInfo
from .frame import ASPFrame
from .header import ASPHeader, ASPFileHeader


__all__ = ['ASPFileReader', 'ASPStreamReader']


class ASPFileReader(FileBase):
    """Simple reader for GUPPI files.

    Wraps a binary filehandle, providing methods to help interpret the data,
    such as `read_frame` and `get_frame_rate`. By default, frame payloads
    are mapped rather than fully read into physical memory.

    Parameters
    ----------
    fh_raw : filehandle
        Filehandle of the raw binary data file.
    """
    info = FileReaderInfo()

    def read_header(self, file_header=None):
        """Read a block header from the file.

        Parameters
        ----------
        file_header : `~baseband.asp.header.ASPFileHeader`, optional
            Possible file header to attach to the block header.

        Returns
        -------
        header : `~baseband.asp.ASPHeader`
        """
        return ASPHeader.fromfile(self.fh_raw, file_header=file_header)

    def read_file_header(self):
        """Read a file header from the file.

        Returns
        -------
        header : `~baseband.asp.ASPFileHeader`
        """
        return ASPFileHeader.fromfile(self.fh_raw)

    def read_frame(self, file_header=None, memmap=False, verify=True):
        """Read the frame header and read or map the corresponding payload.

        Parameters
        ----------
        file_header : `~baseband.asp.header.ASPFileHeader`, optional
            Possible file header to attach to the block header and frame.
        memmap : bool, optional
            If `True`, map the payload using `~numpy.memmap`, so that
            parts are only loaded into memory as needed to access data.
        verify : bool, optional
            Whether to do basic checks of frame integrity.  Default: `True`.

        Returns
        -------
        frame : `~baseband.asp.ASPFrame`
            With ``.header`` and ``.payload`` properties.  The ``.data``
            property returns all data encoded in the frame.  Since this may
            be too large to fit in memory, it may be better to access the
            parts of interest by slicing the frame.
        """
        return ASPFrame.fromfile(self.fh_raw, file_header=file_header,
                                 memmap=memmap, verify=verify)

    def get_frame_rate(self):
        """Determine the number of frames per second.

        The routine uses the sample rate and number of samples per frame
        (excluding overlap) from the first header in the file.

        Returns
        -------
        frame_rate : `~astropy.units.Quantity`
            Frames per second.
        """
        with self.temporary_offset(0):
            file_header = self.read_file_header()
            header = self.read_header(file_header=file_header)
        return (header.sample_rate
                / (header.samples_per_frame - header.overlap)).to(u.Hz)


class ASPStreamReader(StreamReaderBase):

    def __init__(self, fh_raw):
        file_header = ASPFileHeader.fromfile(fh_raw)
        header0 = ASPHeader.fromfile(fh_raw, file_header=file_header)
        super().__init__(
            fh_raw, header0, bps=8, complex_data=True)


open = FileOpener('ASP', header_class=ASPHeader, classes={
    'rb': ASPFileReader,
    'rs': ASPStreamReader}).wrapped(module=__name__, doc="")
