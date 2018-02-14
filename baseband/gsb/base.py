# Licensed under the GPLv3 - see LICENSE.rst
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

__all__ = ['GSBFileReader', 'GSBFileWriter', 'GSBStreamReader',
           'GSBStreamWriter', 'open']


class GSBTimeStampIO(VLBIFileBase):
    """Simple reader/writer for GSB time stamp files.

    Adds ``read_timestamp`` and ``write_timestamp`` methods to the basic VLBI
    file wrapper. To be used with a text file.
    """

    def __init__(self, fh_raw):
        fh_raw = io.TextIOWrapper(fh_raw)
        super(GSBTimeStampIO, self).__init__(fh_raw)

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
            Header holding time to be written to disk.
        **kwargs :
            If no header is given, these are used to initialize one.
        """
        if header is None:
            header = GSBHeader.fromvalues(**kwargs)
        header.tofile(self.fh_raw)


class GSBFileReader(VLBIFileBase):
    """Simple reader for GSB data files.

    Adds ``read_payload`` method to the basic VLBI binary file wrapper.
    """

    def read_payload(self, payloadsize, nchan=1, bps=4, complex_data=False):
        """Read a single block.

        Parameters
        ----------
        payloadsize : int
            Number of bytes to read.
        nchan : int
            Number of channels in the data.  Default: 1.
        bps : int
            Number of bits per sample (or real/imaginary component).
            Default: 4.
        complex_data : bool
            Whether data is complex or float.  Default: False.

        Returns
        -------
        frame : `~baseband.gsb.GSBPayload`
            With a ``.data`` property that returns the data encoded.
        """
        return GSBPayload.fromfile(self.fh_raw, payloadsize=payloadsize,
                                   nchan=nchan, bps=bps,
                                   complex_data=complex_data)


class GSBFileWriter(VLBIFileBase):
    """Simple writer for GSB data files.

    Adds ``write_payload`` method to the basic VLBI binary file wrapper.
    """

    def write_payload(self, data, bps=4):
        """Write single data block.

        Parameters
        ----------
        data : array or :`~baseband.gsb.GSBPayload`
            If an array, ``bps`` needs to be passed in.
        bps : int, optional
            The number of bits per sample to be used to encode the payload.
            Ignored if `data` is a GSB payload.  Default: 4.
        """
        if not isinstance(data, GSBPayload):
            data = GSBPayload.fromdata(data, bps=bps)
        return data.tofile(self.fh_raw)


class GSBStreamBase(VLBIStreamBase):
    """GSB file wrapper, allowing access as a stream of data."""

    _sample_shape_maker = GSBPayload._sample_shape_maker

    def __init__(self, fh_ts, fh_raw, header0, subset=None,
                 nchan=None, bps=None, complex_data=None,
                 samples_per_frame=None, payloadsize=None,
                 sample_rate=None, squeeze=True):
        self.fh_ts = fh_ts
        rawdump = header0.mode == 'rawdump'
        complex_data = (complex_data if complex_data is not None else
                        (False if rawdump else True))
        bps = bps if bps is not None else (4 if rawdump else 8)
        nchan = nchan if nchan is not None else (1 if rawdump else 512)

        # By default, GSB frames span 4 MB for rawdump and 8 MB for phased.
        if payloadsize is None and samples_per_frame is None:
            payloadsize = 2**22 if rawdump else 2**23 // len(fh_raw[0])

        if payloadsize is None:
            payloadsize = (samples_per_frame * nchan *
                           (2 if complex_data else 1) * bps // 8 //
                           (1 if rawdump else len(fh_raw[0])))
        elif samples_per_frame is None:
            samples_per_frame = (payloadsize * 8 // bps *
                                 (1 if rawdump else len(fh_raw[0])) //
                                 (nchan * (2 if complex_data else 1)))

        # By default, GSB rawdump and phased frames span exactly 251.658240 ms.
        if sample_rate is None:
            sample_rate = (samples_per_frame * (100. / 3. / 2.**23)) * u.MHz

        unsliced_shape = (nchan,) if rawdump else (len(fh_raw), nchan)

        super(GSBStreamBase, self).__init__(
            fh_raw, header0=header0, unsliced_shape=unsliced_shape, bps=bps,
            complex_data=complex_data, subset=subset,
            samples_per_frame=samples_per_frame,
            sample_rate=sample_rate, squeeze=squeeze)
        self._payloadsize = payloadsize

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

    Requires both a timestamp and one or more corresponding raw data files.

    Parameters
    ----------
    fh_ts : `~baseband.gsb.base.GSBTimeStampIO` instance
        Header file handle.
    fh_raw : `~io.BufferedReader` instance, or nested tuple of instances
        Raw binary data file handle(s).  A single file is needed for rawdump,
        and a tuple for phased.  For a nested tuple, the outer tuple determines
        the number of polarizations, and the inner tuple(s) the number of
        streams per polarization.  E.g., ((polL1, polL2), (polR1, polR2)) for
        two streams per polarization.  A single tuple is interpreted as
        streams of a single polarization.
    sample_rate : `~astropy.units.Quantity`, optional
        Number of complete samples per second (ie. the rate at which each
        channel of each polarization is sampled).  If not given, will be
        inferred assuming the framerate is exactly 251.658240 ms.
    nchan : int, optional
        Number of channels. Default is `None`, which sets it to 1 for rawdump,
        512 for phased.
    bps : int, optional
        Bits per elementary sample (e.g., the real or imaginary part of each
        complex data sample).  Default is `None`, which sets it to 4 for
        rawdump, 8 for phased.
    complex_data : bool, optional
        Default is `None`, which sets `False` for rawdump, `True` for phased.
    samples_per_frame : int, optional
        Number of complete samples per frame.  Can give ``payloadsize``
        instead.
    payloadsize : int, optional
        Number of bytes per payload, divided by the number of raw files.
        If both ``samples_per_frame`` and ``payloadsize`` are ``None``,
        ``payloadsize`` is set to 2**22 (4 MB) for rawdump, and 2**23 (8 MB)
        divided by the number of streams per polarization for phased.
    subset : indexing object or tuple of objects, optional
        Specific components of the complete sample to decode.  If a single
        indexing object is passed, it selects (available) polarizations.  If a
        tuple of objects is passed, the first selects (available) polarizations
        and the second selects channels.  By default, all components are read.
    squeeze : bool, optional
        If `True` (default), remove any dimensions of length unity from decoded
        data.
    """
    # TODO: right now cannot inherit from GSBFileReader, unlike for other
    # baseband classes, since we need to access multiple files.  Can this
    # be solved with FileWriter/FileReader classes that handle timestamps and
    # multiple blocks, combining these into a frame?

    def __init__(self, fh_ts, fh_raw, sample_rate=None, nchan=None,
                 bps=None, complex_data=None, samples_per_frame=None,
                 payloadsize=None, subset=None, squeeze=True):
        header0 = fh_ts.read_timestamp()
        super(GSBStreamReader, self).__init__(
            fh_ts, fh_raw, header0, nchan=nchan, bps=bps,
            complex_data=complex_data, subset=subset,
            samples_per_frame=samples_per_frame, payloadsize=payloadsize,
            sample_rate=sample_rate, squeeze=squeeze)
        self.fh_ts.seek(0)
        self._frame_nr = None

    @lazyproperty
    def _last_header(self):
        """Last header of the timestamp file."""
        fh_ts_offset = self.fh_ts.tell()
        self.fh_ts.seek(0, 2)
        fh_ts_len = self.fh_ts.tell()
        if fh_ts_len == self.header0.size:
            # Only one line in file
            return self.header0

        # Read last bytes in binary, since cannot seek back from end in
        # text files.
        from_end = min(5 * self.header0.size // 2, fh_ts_len)
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

    def read(self, count=None, fill_value=0., out=None):
        """Read a number of complete (or subset) samples.

        The range retrieved can span multiple frames.

        Parameters
        ----------
        count : int, optional
            Number of complete/subset samples to read.  If omitted or negative,
            the whole file is read.  Ignored if ``out`` is given.
        fill_value : float or complex
            Value to use for invalid or missing data.
        out : `None` or array
            Array to store the data in. If given, ``count`` will be inferred
            from the first dimension.  The other dimensions should equal
            ``sample_shape``.

        Returns
        -------
        out : array of float or complex
            The first dimension is sample-time, and the other two, given by
            ``sample_shape``, are (polarization, channel).  Any
            dimension of length unity is removed if ``self.squeeze=True``.
        """
        if out is None:
            if count is None or count < 0:
                count = self.size - self.offset

            dtype = np.complex64 if self.complex_data else np.float32
            out = np.empty((count,) + self.sample_shape, dtype)
        else:
            assert out.shape[1:] == self.sample_shape, (
                "'out' should have trailing shape {}".format(self.sample_shape))
            count = out.shape[0]

        offset0 = self.offset
        while count > 0:
            frame_nr, sample_offset = divmod(self.offset,
                                             self.samples_per_frame)
            if(frame_nr != self._frame_nr):
                # Read relevant frame (possibly reusing data array from
                # previous frame set).
                self._read_frame(fill_value)
                framerate = self.sample_rate / self.samples_per_frame
                assert np.isclose(self._frame_nr,
                                  ((self._frame.header.time -
                                    self.start_time) * framerate).to(u.one))

            # Determine appropriate slice to decode.
            nsample = min(count, self.samples_per_frame - sample_offset)
            sample = self.offset - offset0
            data_slice = slice(sample_offset, sample_offset + nsample)
            if self.subset:
                data_slice = (data_slice,) + self.subset
            # Copy relevant data from frame into output.
            out[sample:sample + nsample] = (
                self._frame[data_slice].squeeze() if self.squeeze else
                self._frame[data_slice])
            self.offset += nsample
            count -= nsample

        return out

    def _read_frame(self, fill_value=0., out=None):
        frame_nr = self.offset // self.samples_per_frame
        self.fh_ts.seek(self.header0.seek_offset(frame_nr))
        if self.header0.mode == 'rawdump':
            self.fh_raw.seek(frame_nr * self._payloadsize)
        else:
            for fh_pair in self.fh_raw:
                for fh in fh_pair:
                    fh.seek(frame_nr * self._payloadsize)
        self._frame = GSBFrame.fromfile(self.fh_ts, self.fh_raw,
                                        payloadsize=self._payloadsize,
                                        nchan=self._unsliced_shape.nchan,
                                        bps=self.bps,
                                        complex_data=self.complex_data)
        self._frame_nr = frame_nr
        return self._frame


class GSBStreamWriter(GSBStreamBase, VLBIStreamWriterBase):
    """GSB format writer.

    Parameters
    ----------
    fh_ts : `~baseband.gsb.base.GSBTimeStampIO` instance
        Header file handle.
    fh_raw : `~io.BufferedReader` instance, or nested tuple of instances
        Raw binary data file handle(s).  A single file is needed for rawdump,
        and a tuple for phased.  For a nested tuple, the outer tuple determines
        the number of polarizations, and the inner tuple(s) the number of
        streams per polarization.  E.g., ((polL1, polL2), (polR1, polR2)) for
        two streams per polarization.  A single tuple is interpreted as
        streams of a single polarization.
    sample_rate : `~astropy.units.Quantity`, optional
        Number of complete samples per second (ie. the rate at which each
        channel of each polarization is sampled).  If not given, will be
        inferred assuming the framerate is exactly 251.658240 ms.
    header : `~baseband.gsb.GSBHeader`, optional
        Header for the first frame, holding time information, etc.
    nchan : int, optional
        Number of channels. Default is `None`, which sets it to 1 for rawdump,
        512 for phased.
    bps : int, optional
        Bits per elementary sample (e.g., the real or imaginary part of each
        complex data sample).  Default is `None`, which sets it to 4 for
        rawdump, 8 for phased.
    complex_data : bool, optional
        Default is `None`, which sets `False` for rawdump, `True` for phased.
    samples_per_frame : int, optional
        Number of complete samples per frame.  Can give ``payloadsize``
        instead.
    payloadsize : int, optional
        Number of bytes per payload, divided by the number of raw files.
        If both ``samples_per_frame`` and ``payloadsize`` are ``None``,
        ``payloadsize`` is set to 2**22 (4 MB) for rawdump and 2**23 (8 MB)
        divided by the number of streams per polarization for phased.
    squeeze : bool, optional
        If `True` (default), ``write`` accepts squeezed arrays as input, and
        adds any dimensions of length unity.
    **kwargs
        If no header is given, an attempt is made to construct one from
        these.  For a standard header, this would include the following.

    --- Header keywords : (see :meth:`~baseband.vdif.VDIFHeader.fromvalues`)

    header_mode : 'rawdump' or 'phased', optional
        Used to explicitly set the mode of the GSB stream.  If not given, it
        will be 'rawdump' if only a single raw file is present, 'phased'
        otherwise.
    time : `~astropy.time.Time`
        Header time (from the GPS-based timestamp).  (One can alternatively
        pass a string of format ``'YYYY MM DD HH MM SS.SSSSSS'`` to the key
        `gps`.)
    pc_time : `~astropy.time.Time`
        PC-based time; less accurate than GPS time and not used by Baseband.
        (One can alternatively pass a string of format ``'YYYY MM DD HH MM
        SS.SSSSSS'`` to the key `pc`.)
    seq_nr : int
        Frame number.
    mem_block : int
        Redundant modulo-8 shared memory block number; not used by Baseband.
    """

    def __init__(self, fh_ts, fh_raw, sample_rate=None, header=None,
                 nchan=None, bps=None, complex_data=None,
                 samples_per_frame=None, payloadsize=None, squeeze=True,
                 **kwargs):
        if header is None:
            mode = kwargs.pop('header_mode',
                              'rawdump' if hasattr(fh_raw, 'read') else
                              'phased')
            header = GSBHeader.fromvalues(mode=mode, **kwargs)
        super(GSBStreamWriter, self).__init__(
            fh_ts, fh_raw, header, nchan=nchan, bps=bps, subset=None,
            complex_data=complex_data, samples_per_frame=samples_per_frame,
            payloadsize=payloadsize, sample_rate=sample_rate,
            squeeze=squeeze)
        self._data = np.zeros((self.samples_per_frame,) + self._unsliced_shape,
                              (np.complex64 if self.complex_data
                               else np.float32))
        self._valid = True

    def write(self, data):
        """Write data, buffering by frames as needed.

        Parameters
        ----------
        data : array
            Piece of data to be written, with sample dimensions as given by
            ``sample_shape``. This should be properly scaled to make best use
            of the dynamic range delivered by the encoding.
        """
        assert data.shape[1:] == self.sample_shape, (
            "'data' should have trailing shape {}".format(self.sample_shape))
        assert data.dtype.kind == self._data.dtype.kind, (
            "'data' should be {}".format('complex' if self.data.dtype == 'c'
                                         else 'float'))
        if self.squeeze:
            data = self._unsqueeze(data)

        count = data.shape[0]
        sample = 0
        offset0 = self.offset
        while count > 0:
            frame_nr, sample_offset = divmod(self.offset,
                                             self.samples_per_frame)
            if sample_offset == 0:
                # Set up header for new frame.  (mem_block is set to a rotating
                # modulo-8 value with no meaning.)
                time_offset = self.tell(unit=u.s)
                if self.header0.mode == 'phased':
                    self._header = self.header0.__class__.fromvalues(
                        gps_time=self.header0.gps_time + time_offset,
                        pc_time=self.header0.pc_time + time_offset,
                        seq_nr=(frame_nr + self.header0['seq_nr']),
                        mem_block=((self.header0['mem_block'] + frame_nr) % 8))
                else:
                    self._header = self.header0.__class__.fromvalues(
                        time=self.header0.time + time_offset)

            nsample = min(count, self.samples_per_frame - sample_offset)
            sample_end = sample_offset + nsample
            sample = self.offset - offset0
            self._data[sample_offset:sample_end] = data[sample:
                                                        sample + nsample]
            if sample_end == self.samples_per_frame:
                self._frame = GSBFrame.fromdata(self._data, self._header,
                                                self.bps)
                self._frame.tofile(self.fh_ts, self.fh_raw)

            self.offset += nsample
            count -= nsample

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
    When the file is opened as text, one gets a standard file handle, but with
    methods to read/write timestamps.  When it is opened as a binary, one
    similarly gets methods to read/write a frame.  Opened as a stream, the file
    is interpreted as a timestamp file, but raw files need to be given too.
    This allows access to the stream(s) as series of samples.

    Parameters
    ----------
    name : str
        File name of timestamp or raw data file.
    mode : {'rb', 'wb', 'rt', 'wt', 'rs', or 'ws'}, optional
        Whether to open for reading or writing, and as a regular text or binary
        file (for timestamps and data, respectively) or as a stream.  Default
        is `rs` for reading a stream.
    **kwargs
        Additional arguments when opening the file as a stream.

    --- For both reading and writing of streams :

    raw : str or (tuple of) tuple of str
        Name of files holding payload data.  A single file is needed for
        rawdump, and a tuple for phased.  For a nested tuple, the outer tuple
        determines  the number of polarizations, and the inner tuple(s) the
        number of streams per polarization.  E.g.,
        ((polL1, polL2), (polR1, polR2)) for two streams per polarization.  A
        single tuple is interpreted as streams of a single polarization.
    sample_rate : `~astropy.units.Quantity`
        Number of complete samples per second (ie. the rate at which each
        channel of each polarization is sampled).  If not given, will be
        inferred assuming the framerate is exactly 251.658240 ms.
    nchan : int, optional
        Number of channels. Default: 1 for rawdump, 512 for phased.
    bps : int, optional
        Bits per elementary sample (e.g., the real or imaginary part of each
        complex data sample).  Default: 4 for rawdump, 8 for phased.
    complex_data : bool, optional
        Default: `False` for rawdump, `True` for phased.
    samples_per_frame : int
        Number of complete samples per frame.  Can give ``payloadsize``
        instead.
    payloadsize : int, optional
        Number of bytes per payload, divided by the number of raw files.
        If both ``samples_per_frame`` and ``payloadsize`` are ``None``,
        ``payloadsize`` is set to 2**22 (4 MB) for rawdump and 2**23 (8 MB)
        divided by the number of streams per polarization for phased.
    subset : indexing object or tuple of objects, optional
        Specific components of the complete sample to decode.  If a single
        indexing object is passed, it selects (available) polarizations.  If a
        tuple of objects is passed, the first selects (available) polarizations
        and the second selects channels.  By default, all components are read.
    squeeze : bool, optional
        If `True` (default) and reading, remove any dimensions of length unity
        from decoded data.  If `True` and writing, accept squeezed
        arrays as input, and add channel and polarization dimensions if they
        have length unity.

    --- For writing a stream : (see `~baseband.gsb.base.GSBStreamWriter`)

    header : `~baseband.gsb.GSBHeader`
        Header for the first frame, holding time information, etc.
    **kwargs
        If the header is not given, an attempt will be made to construct one
        with any further keyword arguments.  If one requires to explicitly set
        the mode of the GSB stream, use ``header_mode``.  If not given, it
        will be 'rawdump' if only a single raw file is present, 'phased'
        otherwise.  See :class:`~baseband.gsb.base.GSBStreamWriter`.

    Returns
    -------
    Filehandle
        :class:`~baseband.gsb.base.GSBFileReader` or
        :class:`~baseband.gsb.base.GSBFileWriter` instance (binary), or
        :class:`~baseband.gsb.base.GSBStreamReader` or
        :class:`~baseband.gsb.base.GSBStreamWriter` instance (stream)
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
        if kwargs:
            raise TypeError('got unexpected arguments {}'
                            .format(kwargs.keys()))

        opened_files = []
        if not hasattr(name, fh_attr):
            name = io.open(name, mode.replace('t', '').replace('b', '') + 'b')
            opened_files = [name]
        elif isinstance(name, io.TextIOBase):
            raise TypeError("Only binary file handles can be used (even for "
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
