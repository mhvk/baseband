# Licensed under the GPLv3 - see LICENSE
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ..vlbi_base.file_info import VLBIFileReaderInfo


class Mark4FileReaderInfo(VLBIFileReaderInfo):
    """Standardized information on Mark 4 file readers.

    The ``info`` descriptor has a number of standard attributes, which are
    determined from arguments passed in opening the file, from the first header
    (``info.header0``) and from possibly scanning the file to determine the
    duration of frames.  `Mark4FileReaderInfo` has two additional attributes
    specific to Mark 4 files (``ntrack`` and ``offset0``, see below).

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
    missing : dict
        Entries are keyed by names of arguments that should be passed to
        the file reader to obtain full information. The associated entries
        explain why these arguments are needed. For Mark 4, the possible
        entries are ``decade`` and ``ref_time``.

    Examples
    --------
    The most common use is simply to print information::

        >>> from baseband.data import SAMPLE_MARK4
        >>> from baseband import mark4
        >>> fh = mark4.open(SAMPLE_MARK4, 'rb')
        >>> fh.info
        File information:
        format = mark4
        frame_rate = 400.0 Hz
        sample_rate = 32.0 MHz
        samples_per_frame = 80000
        sample_shape = (8,)
        bps = 2
        complex_data = False
        offset0 = 2696
        <BLANKLINE>
        missing:  decade, ref_time: needed to infer full times.
        <BLANKLINE>
        >>> fh.close()

        >>> fh = mark4.open(SAMPLE_MARK4, 'rb', decade=2010)
        >>> fh.info
        File information:
        format = mark4
        frame_rate = 400.0 Hz
        sample_rate = 32.0 MHz
        samples_per_frame = 80000
        sample_shape = (8,)
        bps = 2
        complex_data = False
        start_time = 2014-06-16T07:38:12.475000000
        offset0 = 2696
        >>> fh.close()
    """
    attr_names = VLBIFileReaderInfo.attr_names + ('ntrack', 'offset0')
    _header0_attrs = ('bps', 'samples_per_frame')
    _parent_attrs = ('ntrack', 'decade', 'ref_time')

    def _get_header0(self):
        fh = self._parent
        old_offset = fh.tell()
        try:
            fh.seek(0)
            offset0 = fh.locate_frame()
            if offset0 is None:
                return None

            self.offset0 = offset0
            return fh.read_header()
        except Exception:
            return None
        finally:
            fh.seek(old_offset)

    def _collect_info(self):
        super(Mark4FileReaderInfo, self)._collect_info()
        if self:
            self.complex_data = False
            # TODO: Shouldn't Mark4Header provide this?
            self.sample_shape = (self.header0.nchan,)
            if self.decade is None and self.ref_time is None:
                self.missing['decade'] = self.missing['ref_time'] = (
                    "needed to infer full times.")
