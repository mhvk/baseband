from astropy import units as u

from ..base.base import (FileBase, VLBIFileReaderBase,
                         VLBIStreamReaderBase, FileOpener)
from ..base.file_info import FileReaderInfo
from .frame import ASPFrame
from .header import ASPHeader, ASPFileHeader


__all__ = ['ASPFileReader', 'ASPStreamReader']


class ASPFileReader(VLBIFileReaderBase):
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


class ASPFileWriter(FileBase):
    """Simple writer/mapper for DADA files.

    Adds `write_file_header` and `write_frame` methods to the binary
    file wrapper.
    """

    def write_file_header(self, header=None, **kwargs):
        """Write a single frame (header plus payload).

        Parameters
        ----------
        header : `~baseband.asp.ASPFileHeader`
            Can instead give keyword arguments to construct a header.
            If a `~baseband.asp.ASPHeader`, takes its ``file_header``
            attribute.
        **kwargs
            If ``header`` is not given, these are used to initialize one.
        """
        if isinstance(header, ASPHeader):
            header = header.file_header

        elif header is None:
            header = ASPFileHeader.fromvalues(**kwargs)

        header.tofile(self)

    def write_frame(self, data, header=None, **kwargs):
        """Write a single frame (header plus payload).

        Parameters
        ----------
        data : `~numpy.ndarray` or `~baseband.asp.ASPFrame`
            If an array, a ``header`` should be given, which will be used to
            get the information needed to encode the array, and to construct
            the ASP frame.
        header : `~baseband.asp.ASPHeader`
            Can instead give keyword arguments to construct a header.  Ignored
            if ``data`` is a `~baseband.asp.ASPFrame` instance.
        **kwargs
            If ``header`` is not given, these are used to initialize one.
        """
        if not isinstance(data, ASPFrame):
            data = ASPFrame.fromdata(data, header, **kwargs)
        return data.tofile(self.fh_raw)


class ASPStreamReader(VLBIStreamReaderBase):

    def __init__(self, fh_raw):
        fh_raw = ASPFileReader(fh_raw)
        file_header = fh_raw.read_file_header()
        header0 = fh_raw.read_header(file_header=file_header)
        super().__init__(
            fh_raw, header0, bps=8, complex_data=True)
        # TODO: this would fail with SequentialFile!!
        self._raw_offsets[0] = file_header.nbytes


open = FileOpener('ASP', header_class=ASPHeader, classes={
    'rb': ASPFileReader,
    'wb': ASPFileWriter,
    'rs': ASPStreamReader}).wrapped(module=__name__, doc="")
