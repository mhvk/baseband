# Licensed under the GPLv3 - see LICENSE
import astropy.units as u

from baseband.helpers import sequentialfile as sf
from baseband.base.base import (
    FileBase,
    FileOpener, FileInfo)
from baseband.base.file_info import FileReaderInfo
from .header import KotekanHeader
from .frame import KotekanFrame


__all__ = ['KotekanFileNameSequencer',
           'KotekanFileReader',
           'open', 'info']


class KotekanFileNameSequencer(sf.FileNameSequencer):
    """List-like generator of Kotekan filenames using a template.

    Parameters
    ----------
    template : str
        Template to format to get specific filenames.  Curly bracket item
        keywords are not case-sensitive.
    header : dict-like
        Structure holding key'd values that are used to fill in the format.
        Keys must be in all caps (eg. ``DATE``), as with DADA header keys.
    ndisk : int
        Number of disks involved.

    Examples
    --------

    >>> from baseband import kotekan
    >>> dfs = kotekan.KOTEKANFileNameSequencer(
    ...     '/drives/CHA/{disk}/20210311T000929Z_no_name_set_raw/{file_nr:09d}.raw',
    ...     ndisk=8)
    >>> dfs[10]
    '2018-01-01_010.dada'
    >>> from baseband.data import SAMPLE_DADA
    >>> with open(SAMPLE_DADA, 'rb') as fh:
    ...     header = dada.DADAHeader.fromfile(fh)
    >>> template = '{utc_start}.{obs_offset:016d}.000000.dada'
    >>> dfs = dada.DADAFileNameSequencer(template, header)
    >>> dfs[0]
    '2013-07-02-01:37:40.0000006400000000.000000.dada'
    >>> dfs[1]
    '2013-07-02-01:37:40.0000006400064000.000000.dada'
    >>> dfs[10]
    '2013-07-02-01:37:40.0000006400640000.000000.dada'
    """

    def __init__(self, template, header={}, ndisk=None):
        super().__init__(template, header=header)
        self.ndisk = ndisk
        if ndisk is not None:
            self.items.setdefault('disk', 0)
            self._disk_0 = self.items['disk']

    def _process_items(self, file_nr):
        super()._process_items(file_nr)
        if self.ndisk is not None:
            self.items['disk'] = (self._disk_0
                                  + self.items['file_nr']) % self.ndisk


class KotekanFileReader(FileBase):
    """Simple reader for raw Kotekan files."""
    info = FileReaderInfo()

    def read_header(self):
        """Read a single header from the file.

        Returns
        -------
        header : `~baseband.kotekan.KotekanHeader`
        """
        return KotekanHeader.fromfile(self.fh_raw)

    def read_frame(self, memmap=False, verify=True):
        """Read the frame header and read or map the corresponding payload."""
        return KotekanFrame.fromfile(self.fh_raw, memmap=memmap, verify=verify)

    def get_frame_rate(self):
        """Determine the number of frames per second.

        The routine uses the sample rate and number of samples per frame
        from the first header in the file.

        Returns
        -------
        frame_rate : `~astropy.units.Quantity`
            Frames per second.
        """
        with self.temporary_offset(0):
            header = self.read_header()
        return (header.sample_rate / header.samples_per_frame).to(u.Hz)


class KotekanFileOpener(FileOpener):
    FileNameSequencer = KotekanFileNameSequencer


open = KotekanFileOpener('kotekan', {'rb': KotekanFileReader}, KotekanHeader)
info = FileInfo(open)
