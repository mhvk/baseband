# Licensed under the GPLv3 - see LICENSE
"""The Mark4FileReaderInfo property.

Includes information about what is needed to calcuate times,
number of tracks and offset of first header.
"""

from ..vlbi_base.file_info import VLBIFileReaderInfo, info_item


__all__ = ['Mark4FileReaderInfo']


class Mark4FileReaderInfo(VLBIFileReaderInfo):
    """Standardized information on Mark 4 file readers.

    The ``info`` descriptor has a number of standard attributes, which are
    determined from arguments passed in opening the file, from the first header
    (``info.header0``) and from possibly scanning the file to determine the
    duration of frames.  This class has two additional attributes specific to
    Mark 4 files (``ntrack`` and ``offset0``, see below).

    Attributes
    ----------
    format : str or `None`
        File format, or `None` if the underlying file cannot be parsed.
    frame_rate : `~astropy.units.Quantity`
        Number of data frames per unit of time.
    sample_rate : `~astropy.units.Quantity`
        Complete samples per unit of time.
    samples_per_frame : int
        Number of complete samples in each frame.
    sample_shape : tuple
        Dimensions of each complete sample (e.g., ``(nchan,)``).
    bps : int
        Number of bits used to encode each elementary sample.
    complex_data : bool
        Whether the data are complex.
    start_time : `~astropy.time.Time`
        Time of the first complete sample.
    ntrack : int
        Number of "tape tracks" simulated in the disk file.
    offset0 : int
        Offset in bytes from the start of the file to the location of the
        first header.
    readable : bool
        Whether the first sample could be read and decoded.
    missing : dict
        Entries are keyed by names of arguments that should be passed to
        the file reader to obtain full information. The associated entries
        explain why these arguments are needed. For Mark 4, the possible
        entries are ``decade`` and ``ref_time``.
    errors : dict
        Any exceptions raised while trying to determine attributes.  Keyed
        by the attributes.

    Examples
    --------
    The most common use is simply to print information::

        >>> from baseband.data import SAMPLE_MARK4
        >>> from baseband import mark4
        >>> fh = mark4.open(SAMPLE_MARK4, 'rb')
        >>> fh.info
        File information:
        format = mark4
        number_of_frames = 2
        frame_rate = 400.0 Hz
        sample_rate = 32.0 MHz
        samples_per_frame = 80000
        sample_shape = (8,)
        bps = 2
        complex_data = False
        readable = True
        ntrack = 64
        offset0 = 2696
        <BLANKLINE>
        missing:  decade, ref_time: needed to infer full times.
        <BLANKLINE>
        checks:  decodable: True
        >>> fh.close()

        >>> fh = mark4.open(SAMPLE_MARK4, 'rb', decade=2010)
        >>> fh.info
        File information:
        format = mark4
        number_of_frames = 2
        frame_rate = 400.0 Hz
        sample_rate = 32.0 MHz
        samples_per_frame = 80000
        sample_shape = (8,)
        bps = 2
        complex_data = False
        start_time = 2014-06-16T07:38:12.475000000
        readable = True
        ntrack = 64
        offset0 = 2696
        <BLANKLINE>
        checks:  decodable: True
        >>> fh.close()
    """
    attr_names = (VLBIFileReaderInfo.attr_names[:-4]
                  + ('ntrack', 'offset0')
                  + VLBIFileReaderInfo.attr_names[-4:])
    """Attributes that the container provides."""

    _header0_attrs = ('bps', 'samples_per_frame')
    _parent_attrs = ('ntrack', 'decade', 'ref_time')

    @info_item
    def time_info(self):
        """Additional time info needed to get the start time."""
        time_info = (self.decade, self.ref_time)
        if time_info == (None, None):
            self.missing['decade'] = self.missing['ref_time'] = (
                "needed to infer full times.")
            return None

        return time_info

    @info_item
    def offset0(self):
        """Offset in bytes to the location of the first header."""
        with self._parent.temporary_offset() as fh:
            fh.seek(0)
            fh.find_header()
            return fh.tell()

    @info_item(needs='offset0')
    def header0(self):
        with self._parent.temporary_offset() as fh:
            fh.seek(self.offset0)
            return fh.read_header()

    @info_item(needs='header0')
    def frame0(self):
        with self._parent.temporary_offset() as fh:
            fh.seek(self.offset0)
            return fh.read_frame()

    @info_item(needs='header0')
    def number_of_frames(self):
        """Total number of frames."""
        with self._parent.temporary_offset() as fh:
            fh.seek(-self.header0.frame_nbytes, 2)
            fh.find_header(self.header0, forward=False)
            number_of_frames = ((fh.tell() - self.offset0)
                                / self.header0.frame_nbytes) + 1

        if number_of_frames % 1 == 0:
            return int(number_of_frames)
        else:
            self.warnings['number_of_frames'] = (
                'file contains non-integer number ({}) of frames'
                .format(number_of_frames))
            return None

    complex_data = False

    @info_item(needs='header0')
    def sample_shape(self):
        """Dimensions of each complete sample."""
        return (self.header0.nchan,)

    # Override just to replace what it "needs".
    @info_item(needs=('header0', 'time_info'))
    def start_time(self):
        """Time of the first sample."""
        return super().start_time
