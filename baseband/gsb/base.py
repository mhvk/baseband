# Licensed under the GPLv3 - see LICENSE
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import io
import numpy as np
from astropy.utils import lazyproperty
import astropy.units as u
import warnings
from ..vlbi_base.base import (VLBIFileBase, VLBIStreamBase,
                              VLBIStreamReaderBase, VLBIStreamWriterBase)
from .header import GSBHeader
from .payload import GSBPayload
from .frame import GSBFrame
from .file_info import GSBTimeStampInfo, GSBStreamReaderInfo


__all__ = ['GSBFileReader', 'GSBFileWriter', 'GSBStreamReader',
           'GSBStreamWriter', 'open']


class GSBTimeStampIO(VLBIFileBase):
    """Simple reader/writer for GSB time stamp files.

    Wraps a binary filehandle, providing methods `read_timestamp`,
    `write_timestamp`, and `get_frame_rate`.

    Parameters
    ----------
    fh_raw : filehandle
        Filehandle to the timestamp file, opened in binary mode.
    """

    def __init__(self, fh_raw):
        fh_raw = io.TextIOWrapper(fh_raw)
        super(GSBTimeStampIO, self).__init__(fh_raw)

    info = GSBTimeStampInfo()

    def read_timestamp(self):
        """Read a single timestamp.

        Returns
        -------
        frame : `~baseband.gsb.GSBHeader`
            With a ``.time`` property that returns the time encoded.
        """
        return GSBHeader.fromfile(self.fh_raw)

    def write_timestamp(self, header=None, **kwargs):
        """Write a single timestamp.

        Parameters
        ----------
        header : `~baseband.gsb.GSBHeader`, optional
            Header holding time to be written to disk.  Can instead give
            keyword arguments to construct a header.
        **kwargs :
            If ``header`` is not given, these are used to initialize one.
        """
        if header is None:
            header = GSBHeader.fromvalues(**kwargs)
        header.tofile(self.fh_raw)

    def get_frame_rate(self):
        """Determine the number of frames per second.

        The frame rate is inferred from the first two timestamps.

        Returns
        -------
        frame_rate : `~astropy.units.Quantity`
            Frames per second.
        """
        oldpos = self.tell()
        self.seek(0)
        try:
            timestamp0 = self.read_timestamp()
            timestamp1 = self.read_timestamp()
            return (1. / (timestamp1.time - timestamp0.time)).to(u.Hz)
        finally:
            self.seek(oldpos)


class GSBFileReader(VLBIFileBase):
    """Simple reader for GSB data files.

    Wraps a binary filehandle, providing a `read_payload` method to help
    interpret the data.

    Parameters
    ----------
    payload_nbytes : int
        Number of bytes to read.
    nchan : int, optional
        Number of channels.  Default: 1.
    bps : int, optional
        Bits per elementary sample, i.e. per real or imaginary component
        for complex data.  Default: 4.
    complex_data : bool, optional
        Whether data are complex.  Default: `False`.
    """
    def __init__(self, fh_raw, payload_nbytes, nchan=1, bps=4,
                 complex_data=False):
        self.payload_nbytes = payload_nbytes
        self.nchan = nchan
        self.bps = bps
        self.complex_data = complex_data
        super(GSBFileReader, self).__init__(fh_raw)

    def __repr__(self):
        return ("{name}(fh_raw={s.fh_raw}, payload_nbytes={s.payload_nbytes}, "
                "nchan={s.nchan}, bps={s.bps}, complex_data={s.complex_data})"
                .format(name=self.__class__.__name__, s=self))

    def read_payload(self):
        """Read a single block.

        Returns
        -------
        frame : `~baseband.gsb.GSBPayload`
            With a ``.data`` property that returns the data encoded.
        """
        return GSBPayload.fromfile(self.fh_raw,
                                   payload_nbytes=self.payload_nbytes,
                                   nchan=self.nchan, bps=self.bps,
                                   complex_data=self.complex_data)


class GSBFileWriter(VLBIFileBase):
    """Simple writer for GSB data files.

    Adds `write_payload` method to the basic VLBI binary file wrapper.
    """

    def write_payload(self, data, bps=4):
        """Write single data block.

        Parameters
        ----------
        data : `~numpy.ndarray` or `~baseband.gsb.GSBPayload`
            If an array, ``bps`` needs to be passed in.
        bps : int, optional
            Bits per elementary sample, to use when encoding the payload.
            Ignored if ``data`` is a GSB payload.  Default: 4.
        """
        if not isinstance(data, GSBPayload):
            data = GSBPayload.fromdata(data, bps=bps)
        return data.tofile(self.fh_raw)


class GSBStreamBase(VLBIStreamBase):
    """Base for GSB streams."""

    _sample_shape_maker = GSBPayload._sample_shape_maker

    def __init__(self, fh_ts, fh_raw, header0, sample_rate=None,
                 samples_per_frame=None, payload_nbytes=None, nchan=None,
                 bps=None, complex_data=None, squeeze=True, subset=(),
                 verify=True):

        self.fh_ts = fh_ts
        rawdump = header0.mode == 'rawdump'
        complex_data = (complex_data if complex_data is not None else
                        (False if rawdump else True))
        bps = bps if bps is not None else (4 if rawdump else 8)
        nchan = nchan if nchan is not None else (1 if rawdump else 512)

        # By default, GSB frames span 4 MB for rawdump and 8 MB for phased.
        if payload_nbytes is None and samples_per_frame is None:
            payload_nbytes = 2**22 if rawdump else 2**23 // len(fh_raw[0])

        if payload_nbytes is None:
            payload_nbytes = (samples_per_frame * nchan *
                              (2 if complex_data else 1) * bps // 8 //
                              (1 if rawdump else len(fh_raw[0])))
        elif samples_per_frame is None:
            samples_per_frame = (payload_nbytes * 8 // bps *
                                 (1 if rawdump else len(fh_raw[0])) //
                                 (nchan * (2 if complex_data else 1)))

        # By default, GSB rawdump and phased frames span exactly 0.251658240 s.
        if sample_rate is None:
            sample_rate = (samples_per_frame * (100. / 3. / 2.**23)) * u.MHz

        unsliced_shape = (nchan,) if rawdump else (len(fh_raw), nchan)

        super(GSBStreamBase, self).__init__(
            fh_raw, header0, sample_rate=sample_rate,
            samples_per_frame=samples_per_frame, unsliced_shape=unsliced_shape,
            bps=bps, complex_data=complex_data, squeeze=squeeze, subset=subset,
            fill_value=0., verify=verify)

        self._payload_nbytes = payload_nbytes

    def close(self):
        self.fh_ts.close()
        try:
            self.fh_raw.close()
        except AttributeError:
            for fh_pair in self.fh_raw:
                for fh in fh_pair:
                    fh.close()

    def __repr__(self):
        if isinstance(self.fh_raw, (list, tuple)):
            data_name = tuple(tuple(p.name.split('/')[-1] for p in pol)
                              for pol in self.fh_raw)
        else:
            data_name = self.fh_raw.name
        return ("<{s.__class__.__name__} header={s.fh_ts.name}"
                " offset= {s.offset}\n    data={dn}\n"
                "    sample_rate={s.sample_rate:.5g},"
                " samples_per_frame={s.samples_per_frame},\n"
                "    sample_shape={s.sample_shape}, bps={s.bps},\n"
                "    {sub}start_time={s.start_time.isot}>"
                .format(s=self, dn=data_name, sub=(
                    'subset={0}, '.format(self.subset) if
                    self.subset else '')))


class GSBStreamReader(GSBStreamBase, VLBIStreamReaderBase):
    """GSB format reader.

    Allows access to GSB files as a continuous series of samples.  Requires
    both a timestamp and one or more corresponding raw data files.

    Parameters
    ----------
    fh_ts : `~baseband.gsb.base.GSBTimeStampIO`
        Header filehandle.
    fh_raw : filehandle, or nested tuple of filehandles
        Raw binary data filehandle(s).  A single file is needed for rawdump,
        and a tuple for phased.  For a nested tuple, the outer tuple determines
        the number of polarizations, and the inner tuple(s) the number of
        streams per polarization.  E.g., ``((polL1, polL2), (polR1, polR2))``
        for two streams per polarization.  A single tuple is interpreted as
        streams of a single polarization.
    sample_rate : `~astropy.units.Quantity`, optional
        Number of complete samples per second, i.e. the rate at which each
        channel of each polarization is sampled.  If `None`, will be
        inferred assuming the frame rate is exactly 0.25165824 s.
    samples_per_frame : int, optional
        Number of complete samples per frame.  Can give ``payload_nbytes``
        instead.
    payload_nbytes : int, optional
        Number of bytes per payload, divided by the number of raw files.
        If both ``samples_per_frame`` and ``payload_nbytes`` are `None`,
        ``payload_nbytes`` is set to ``2**22`` (4 MB) for rawdump, and
        ``2**23`` (8 MB) divided by the number of streams per polarization for
        phased.
    nchan : int, optional
        Number of channels. Default: 1 for rawdump, 512 for phased.
    bps : int, optional
        Bits per elementary sample, i.e. per real or imaginary component for
        complex data.  Default: 4 for rawdump, 8 for phased.
    complex_data : bool, optional
        Whether data are complex.  Default: `False` for rawdump, `True` for
        phased.
    squeeze : bool, optional
        If `True` (default), remove any dimensions of length unity from decoded
        data.
    subset : indexing object or tuple of objects, optional
        Specific components of the complete sample to decode (after possibly
        squeezing).  If a single indexing object is passed, it selects
        (available) polarizations.  If a tuple is passed, the first selects
        polarizations and the second selects channels.  If the tuple is empty
        (default), all components are read.
    verify : bool, optional
        Whether to do basic checks of frame integrity when reading.  The first
        frame of the stream is always checked.  Default: `True`.
    """
    # TODO: right now cannot inherit from GSBFileReader, unlike for other
    # baseband classes, since we need to access multiple files.  Can this
    # be solved with FileWriter/FileReader classes that handle timestamps and
    # multiple blocks, combining these into a frame?

    def __init__(self, fh_ts, fh_raw, sample_rate=None, samples_per_frame=None,
                 payload_nbytes=None, nchan=None, bps=None, complex_data=None,
                 squeeze=True, subset=(), verify=True):
        header0 = fh_ts.read_timestamp()
        super(GSBStreamReader, self).__init__(
            fh_ts, fh_raw, header0, sample_rate=sample_rate,
            samples_per_frame=samples_per_frame, payload_nbytes=payload_nbytes,
            nchan=nchan, bps=bps, complex_data=complex_data,
            squeeze=squeeze, subset=subset, verify=verify)
        self.fh_ts.seek(0)

    info = GSBStreamReaderInfo()

    @lazyproperty
    def _last_header(self):
        """Last header of the timestamp file."""
        fh_ts_offset = self.fh_ts.tell()
        self.fh_ts.seek(0, 2)
        fh_ts_len = self.fh_ts.tell()
        if fh_ts_len == self.header0.nbytes:
            # Only one line in file
            return self.header0

        # Read last bytes in binary, since cannot seek back from end in
        # text files.
        from_end = min(5 * self.header0.nbytes // 2, fh_ts_len)
        self.fh_ts.buffer.seek(-from_end, 2)
        last_lines = self.fh_ts.buffer.read(from_end).strip().split(b'\n')
        self.fh_ts.seek(fh_ts_offset)
        last_line = last_lines[-1].decode('ascii')
        last_line_tuple = tuple(last_line.split())
        # If the last header is missing characters, use the header before it
        # (which may be the first header).
        try:
            assert (len(" ".join(last_line_tuple)) >=
                    len(" ".join(self.header0.words)))
            last_header = self.header0.__class__(last_line_tuple)
        except Exception:
            warnings.warn("The last header entry, '{0}', has an incorect "
                          "length.  Using the second-to-last entry instead."
                          .format(last_line))
            second_last_line = last_lines[-2].decode('ascii')
            second_last_line_tuple = tuple(second_last_line.split())
            last_header = self.header0.__class__(second_last_line_tuple)
        return last_header

    def _read_frame(self, index):
        self.fh_ts.seek(self.header0.seek_offset(index))
        if self.header0.mode == 'rawdump':
            self.fh_raw.seek(index * self._payload_nbytes)
        else:
            for fh_pair in self.fh_raw:
                for fh in fh_pair:
                    fh.seek(index * self._payload_nbytes)
        frame = GSBFrame.fromfile(self.fh_ts, self.fh_raw,
                                  payload_nbytes=self._payload_nbytes,
                                  nchan=self._unsliced_shape.nchan,
                                  bps=self.bps, complex_data=self.complex_data,
                                  verify=self.verify)
        assert int(round(((frame.header.time - self.start_time) *
                          self.sample_rate / self.samples_per_frame)
                         .to_value(u.one))) == index
        return frame


class GSBStreamWriter(GSBStreamBase, VLBIStreamWriterBase):
    """GSB format writer.

    Encodes and writes sequences of samples to file.

    Parameters
    ----------
    fh_ts : `~baseband.gsb.base.GSBTimeStampIO`
        For writing headers to storage.
    fh_raw : filehandle, or nested tuple of filehandles
        For writing raw binary data to storage.  A single file is needed for
        rawdump, and a tuple for phased.  For a nested tuple, the outer
        tuple determines the number of polarizations, and the inner tuple(s)
        the number of streams per polarization.  E.g., ``((polL1, polL2),
        (polR1, polR2))`` for two streams per polarization.  A single tuple is
        interpreted as streams of a single polarization.
    header0 : `~baseband.gsb.GSBHeader`
        Header for the first frame, holding time information, etc.  Can instead
        give keyword arguments to construct a header (see ``**kwargs``).
    sample_rate : `~astropy.units.Quantity`, optional
        Number of complete samples per second, i.e. the rate at which each
        channel of each polarization is sampled.  If not given, will be
        inferred assuming the frame rate is exactly 0.25165824 s.
    samples_per_frame : int, optional
        Number of complete samples per frame.  Can give ``payload_nbytes``
        instead.
    payload_nbytes : int, optional
        Number of bytes per payload, divided by the number of raw files.
        If both ``samples_per_frame`` and ``payload_nbytes`` are `None`,
        ``payload_nbytes`` is set to ``2**22`` (4 MB) for rawdump, and
        ``2**23`` (8 MB) divided by the number of streams per polarization for
        phased.
    nchan : int, optional
        Number of channels. Default: 1 for rawdump, 512 for phased.
    bps : int, optional
        Bits per elementary sample, i.e. per real or imaginary component for
        complex data.  Default: 4 for rawdump, 8 for phased.
    complex_data : bool, optional
        Whether data are complex.  Default: `False` for rawdump, `True` for
        phased.
    squeeze : bool, optional
        If `True` (default), `write` accepts squeezed arrays as input, and
        adds any dimensions of length unity.
    **kwargs
        If no header is given, an attempt is made to construct one from these.
        For a standard header, this would include the following.

    --- Header keywords : (see :meth:`~baseband.gsb.GSBHeader.fromvalues`)

    time : `~astropy.time.Time`
        Start time of the file.
    header_mode : 'rawdump' or 'phased', optional
        Used to explicitly set the mode of the GSB stream.  Default: 'rawdump'
        if only a single raw file is present, or 'phased' otherwise.
    seq_nr : int, optional
        Frame number, only used for phased (default: 0).
    """

    def __init__(self, fh_ts, fh_raw, header0=None, sample_rate=None,
                 samples_per_frame=None, payload_nbytes=None, nchan=None,
                 bps=None, complex_data=None, squeeze=True, **kwargs):
        if header0 is None:
            mode = kwargs.pop('header_mode',
                              'rawdump' if hasattr(fh_raw, 'read') else
                              'phased')
            header0 = GSBHeader.fromvalues(mode=mode, **kwargs)
        super(GSBStreamWriter, self).__init__(
            fh_ts, fh_raw, header0, sample_rate=sample_rate,
            samples_per_frame=samples_per_frame, payload_nbytes=payload_nbytes,
            nchan=nchan, bps=bps, complex_data=complex_data, squeeze=squeeze)
        self._payload = GSBPayload.fromdata(
            np.zeros((self.samples_per_frame,) + self._unsliced_shape,
                     (np.complex64 if self.complex_data else np.float32)),
            bps=self.bps)

    def _make_frame(self, index):
        # Set up header for new frame.  (mem_block is set to a rotating
        # modulo-8 value with no meaning.)
        time_offset = index * self.samples_per_frame / self.sample_rate
        if self.header0.mode == 'phased':
            header = self.header0.fromvalues(
                gps_time=self.header0.gps_time + time_offset,
                pc_time=self.header0.pc_time + time_offset,
                seq_nr=(index + self.header0['seq_nr']),
                mem_block=((self.header0['mem_block'] + index) % 8))
        else:
            header = self.header0.fromvalues(time=self.start_time +
                                             time_offset)

        return GSBFrame(header, self._payload, valid=True, verify=False)

    def _write_frame(self, frame, valid=True):
        assert valid
        frame.tofile(self.fh_ts, self.fh_raw)

    def flush(self):
        self.fh_ts.flush()
        try:
            self.fh_raw.flush()
        except AttributeError:
            for fh_pair in self.fh_raw:
                for fh in fh_pair:
                    fh.flush()


def open(name, mode='rs', **kwargs):
    """Open GSB file(s) for reading or writing.

    A GSB data set contains a text header file and one or more raw data files.
    When the file is opened as text, one gets a standard filehandle, but with
    methods to read/write timestamps.  When it is opened as a binary, one
    similarly gets methods to read/write frames.  Opened as a stream, the file
    is interpreted as a timestamp file, but raw files need to be given too.
    This allows access to the stream(s) as series of samples.

    Parameters
    ----------
    name : str
        Filename of timestamp or raw data file.
    mode : {'rb', 'wb', 'rt', 'wt', 'rs', or 'ws'}, optional
        Whether to open for reading or writing, and as a regular text or binary
        file (for timestamps and data, respectively) or as a stream.
        Default: 'rs', for reading a stream.
    **kwargs
        Additional arguments when opening the file as a stream.

    --- For both reading and writing of streams :

    raw : str or (tuple of) tuple of str
        Name of files holding payload data.  A single file is needed for
        rawdump, and a tuple for phased.  For a nested tuple, the outer tuple
        determines the number of polarizations, and the inner tuple(s) the
        number of streams per polarization.  E.g.,  ``((polL1, polL2),
        (polR1, polR2))`` for two streams per polarization.  A
        single tuple is interpreted as streams of a single polarization.
    sample_rate : `~astropy.units.Quantity`, optional
        Number of complete samples per second, i.e. the rate at which each
        channel of each polarization is sampled.  If `None`, will be
        inferred assuming the frame rate is exactly 251.658240 ms.
    samples_per_frame : int, optional
        Number of complete samples per frame.  Can give ``payload_nbytes``
        instead.
    payload_nbytes : int, optional
        Number of bytes per payload, divided by the number of raw files.
        If both ``samples_per_frame`` and ``payload_nbytes`` are `None`,
        ``payload_nbytes`` is set to ``2**22`` (4 MB) for rawdump, and
        ``2**23`` (8 MB) divided by the number of streams per polarization for
        phased.
    nchan : int, optional
        Number of channels. Default: 1 for rawdump, 512 for phased.
    bps : int, optional
        Bits per elementary sample, i.e. per real or imaginary component for
        complex data.  Default: 4 for rawdump, 8 for phased.
    complex_data : bool, optional
        Whether data are complex.  Default: `False` for rawdump, `True` for
        phased.
    squeeze : bool, optional
        If `True` (default) and reading, remove any dimensions of length unity
        from decoded data.  If `True` and writing, accept squeezed arrays as
        input, and adds any dimensions of length unity.

    --- For reading only :  (see `~baseband.gsb.base.GSBStreamReader`)

    subset : indexing object or tuple of objects, optional
        Specific components of the complete sample to decode (after possibly
        squeezing).  If a single indexing object is passed, it selects
        (available) polarizations.  If a tuple is passed, the first selects
        polarizations and the second selects channels.  If the tuple is empty
        (default), all components are read.
    verify : bool, optional
        Whether to do basic checks of frame integrity when reading.  The first
        frame of the stream is always checked.  Default: `True`.

    --- For writing only : (see `~baseband.gsb.base.GSBStreamWriter`)

    header0 : `~baseband.gsb.GSBHeader`
        Header for the first frame, holding time information, etc.  Can instead
        give keyword arguments to construct a header.
    **kwargs
        If the header is not given, an attempt will be made to construct one
        with any further keyword arguments.  If one requires to explicitly set
        the mode of the GSB stream, use ``header_mode``.  If not given, it
        will be 'rawdump' if only a single raw file is present, or 'phased'
        otherwise.  See :class:`~baseband.gsb.base.GSBStreamWriter`.

    Returns
    -------
    Filehandle
        :class:`~baseband.gsb.base.GSBFileReader` or
        :class:`~baseband.gsb.base.GSBFileWriter` (binary), or
        :class:`~baseband.gsb.base.GSBStreamReader` or
        :class:`~baseband.gsb.base.GSBStreamWriter` (stream)
    """
    # TODO: think whether the inheritance of StreamReader from FileReader
    # can be made to work (or from TimeStampIO?).
    # TODO: this partially replicates the default opener in vlbi_base;
    # can some parts be factored out?
    if not ('r' in mode or 'w' in mode):
        raise ValueError("Only support opening GSB file for reading "
                         "or writing (mode='r' or 'w').")
    fh_attr = 'read' if 'r' in mode else 'write'
    if 't' in mode or 'b' in mode:
        opened_files = []
        if not hasattr(name, fh_attr):
            name = io.open(name, mode.replace('t', '').replace('b', '') + 'b')
            opened_files = [name]
        elif isinstance(name, io.TextIOBase):
            raise TypeError("Only binary filehandles can be used (even for "
                            "for timestamp files).")
        if 't' in mode:
            cls = GSBTimeStampIO
        else:
            cls = GSBFileWriter if 'w' in mode else GSBFileReader
    else:
        # stream mode.
        name = open(name, mode.replace('s', '') + 't')
        opened_files = [name]
        # Single or multiple files.
        raw = kwargs.pop('raw')
        if not isinstance(raw, (list, tuple)):
            if hasattr(raw, fh_attr):
                fh_raw = raw
            else:
                fh_raw = io.open(raw, mode.replace('s', '') + 'b')
                opened_files.append(raw)
        else:
            if not isinstance(raw[0], (list, tuple)):
                raw = (raw,)
            fh_raw = []
            for pol in raw:
                raw_pol = []
                for p in pol:
                    if hasattr(p, fh_attr):
                        raw_pol.append(p)
                    else:
                        raw_pol.append(io.open(p, mode.replace('s', '') + 'b'))
                        opened_files.append(p)
                fh_raw.append(raw_pol)

        kwargs['fh_raw'] = fh_raw
        cls = GSBStreamWriter if 'w' in mode else GSBStreamReader

    try:
        return cls(name, **kwargs)
    except Exception as exc:
        if opened_files:
            try:
                for name in opened_files:
                    name.close()
            except Exception:  # pragma: no cover
                pass
        raise exc
