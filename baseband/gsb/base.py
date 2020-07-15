# Licensed under the GPLv3 - see LICENSE
import warnings

import numpy as np
import astropy.units as u
from astropy.utils import lazyproperty

from ..base.base import (
    FileBase,
    StreamBase, StreamReaderBase, StreamWriterBase,
    FileOpener, FileInfo)
from .header import GSBHeader
from .payload import GSBPayload
from .frame import GSBFrame
from .file_info import GSBTimeStampInfo, GSBStreamReaderInfo


__all__ = ['GSBTimeStampIO', 'GSBFileReader', 'GSBFileWriter',
           'GSBStreamBase', 'GSBStreamReader', 'GSBStreamWriter',
           'open', 'info']


class GSBTimeStampIO(FileBase):
    """Simple reader/writer for GSB time stamp files.

    Wraps a binary filehandle, providing methods `read_timestamp`,
    `write_timestamp`, and `get_frame_rate`.

    Parameters
    ----------
    fh_raw : filehandle
        Filehandle to the timestamp file, opened in binary mode.
    """

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
        with self.temporary_offset(0):
            timestamp0 = self.read_timestamp()
            timestamp1 = self.read_timestamp()
        return (1. / (timestamp1.time - timestamp0.time)).to(u.Hz)


class GSBFileReader(FileBase):
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
        super().__init__(fh_raw)

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
                                   sample_shape=(self.nchan,), bps=self.bps,
                                   complex_data=self.complex_data)


class GSBFileWriter(FileBase):
    """Simple writer for GSB data files.

    Adds `write_payload` method to the basic binary file wrapper.
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


class GSBStreamBase(StreamBase):
    """Base for GSB streams."""

    _sample_shape_maker = GSBPayload._sample_shape_maker

    def __init__(self, fh_ts, fh_raw, header0, sample_rate=None,
                 samples_per_frame=None, payload_nbytes=None, nchan=None,
                 bps=None, complex_data=None, **kwargs):

        self.fh_ts = fh_ts
        rawdump = header0.mode == 'rawdump'
        if isinstance(fh_raw, (tuple, list)):
            assert not rawdump
            for pair in fh_raw:
                assert isinstance(pair, (tuple, list))
                assert len(pair) == len(fh_raw[0])
        elif not rawdump:
            fh_raw = ((fh_raw,),)

        complex_data = (complex_data if complex_data is not None else
                        (False if rawdump else True))
        bps = bps if bps is not None else (4 if rawdump else 8)
        nchan = nchan if nchan is not None else (1 if rawdump else 512)
        bpfs = bps * nchan * (2 if complex_data else 1)
        default_frame_rate = (1e8/6/2**22)*u.Hz
        nfiles = 1 if rawdump else len(fh_raw[0])
        # By default, GSB payloads are always 4 MB (which can combine to
        # frames of 8 MB for phased if two streams are used).
        if payload_nbytes is None:
            if samples_per_frame is None:
                if sample_rate is None:
                    payload_nbytes = 2**22
                else:
                    payload_nbytes = int((sample_rate / default_frame_rate
                                          * bpfs / 8 / nfiles)
                                         .to(u.one).round())
            else:
                payload_nbytes = samples_per_frame * bpfs // (8 * nfiles)

        if samples_per_frame is None:
            samples_per_frame = payload_nbytes * 8 // bpfs * nfiles
        elif samples_per_frame != payload_nbytes*nfiles*8/bpfs:
            raise ValueError('inconsistent samples_per_frame, bps, '
                             'complex_data, and payload_nbytes')

        if sample_rate is None:
            sample_rate = samples_per_frame * default_frame_rate

        sample_shape = (nchan,) if rawdump else (len(fh_raw), nchan)

        super().__init__(
            fh_raw, header0, sample_rate=sample_rate,
            samples_per_frame=samples_per_frame, sample_shape=sample_shape,
            bps=bps, complex_data=complex_data, **kwargs)

        self._payload_nbytes = payload_nbytes

    @property
    def payload_nbytes(self):
        """Number of bytes per payload, divided by the number of raw files."""
        return self._payload_nbytes

    def __getattr__(self, attr):
        """Try to get things on the current open file if it is not on self."""
        if attr in {'readable', 'writable', 'seekable', 'closed', 'name'}:
            fh_raw = (self.fh_raw if self.header0.mode == 'rawdump'
                      else self.fh_raw[0][0])
            try:
                return getattr(fh_raw, attr)
            except AttributeError:  # pragma: no cover
                pass
        #  __getattribute__ to raise appropriate error.
        return self.__getattribute__(attr)

    def _set_index(self, header, index):
        if self.header0.mode == 'phased':
            time_offset = index / self._frame_rate
            # mem_block is a rotating modulo-8 value with no meaning.
            header.update(gps_time=self.header0.gps_time + time_offset,
                          pc_time=self.header0.pc_time + time_offset,
                          seq_nr=self.header0['seq_nr'] + index,
                          mem_block=(self.header0['mem_block'] + index) % 8)
        else:
            super()._set_index(header, index)

    def close(self):
        self.fh_ts.close()
        if self.header0.mode == 'rawdump':
            self.fh_raw.close()
        else:
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


class GSBStreamReader(GSBStreamBase, StreamReaderBase):
    """GSB format reader.

    Allows access to GSB files as a continuous series of samples.  Requires
    both a timestamp and one or more corresponding raw data files.

    Parameters
    ----------
    fh_ts : filehandle
        For reading timestamps.
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
        Number of complete samples per frame (possibly combining two files).
        Can give ``payload_nbytes`` instead.
    payload_nbytes : int, optional
        Number of bytes per payload (in each raw file separately).
        If both ``samples_per_frame`` and ``payload_nbytes`` are `None`,
        ``payload_nbytes`` is set to ``2**22`` (4 MiB).
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
    # TODO: right we are not really compatible with StreamReaderBase,
    # since we need to access multiple files.  Can this be solved with
    # FileWriter/FileReader classes that handle timestamps and multiple blocks,
    # combining these into a frame?
    def __init__(self, fh_ts, fh_raw, sample_rate=None, samples_per_frame=None,
                 payload_nbytes=None, nchan=None, bps=None, complex_data=None,
                 squeeze=True, subset=(), verify=True):
        fh_ts = GSBTimeStampIO(fh_ts)
        header0 = fh_ts.read_timestamp()
        super().__init__(
            fh_ts, fh_raw, header0, sample_rate=sample_rate,
            samples_per_frame=samples_per_frame, payload_nbytes=payload_nbytes,
            nchan=nchan, bps=bps, complex_data=complex_data,
            squeeze=squeeze, subset=subset, verify=verify)
        self.fh_ts.seek(0)
        # Replace fh_raw with GSBFileReader instances.
        fr_kwargs = dict(payload_nbytes=self._payload_nbytes,
                         nchan=self._unsliced_shape.nchan,
                         bps=self.bps, complex_data=self.complex_data)
        if header0.mode == 'rawdump':
            self.fh_raw = GSBFileReader(self.fh_raw, **fr_kwargs)
        else:
            self.fh_raw = [[GSBFileReader(fh, **fr_kwargs)
                            for fh in fh_pair] for fh_pair in self.fh_raw]

    info = GSBStreamReaderInfo()

    @lazyproperty
    def _last_header(self):
        """Last header of the timestamp file."""
        with self.fh_ts.temporary_offset() as fh:
            # Guess based on a fixed header size.  In reality, this
            # may be an overestimate as the headers can grow in size,
            # or an underestimate as the last header may be partial.
            # So, search around to be sure.
            fh_size = fh.seek(0, 2)
            guess = max(fh_size // self.header0.nbytes, 1)
            while self.header0.seek_offset(guess) > fh_size:
                guess -= 1
            while self.header0.seek_offset(guess) < fh_size:
                guess += 1

            # Now see if there is indeed a nice header before.
            fh.seek(self.header0.seek_offset(guess-1))
            last_line = fh.readline()
            last_line_tuple = last_line.split()
            # But realize that sometimes an incomplete header is written.
            try:
                if (len(" ".join(last_line_tuple))
                        < len(" ".join(self.header0.words))):
                    raise EOFError
                last_header = self.header0.__class__(last_line_tuple)
                # Check header time can be parsed.
                last_header.time
            except Exception:
                warnings.warn("The last header entry, '{0}', has an incorect "
                              "length. Using the second-to-last entry instead."
                              .format(last_line))
                fh.seek(self.header0.seek_offset(guess-2))
                last_line_tuple = fh.readline().split()
                last_header = self.header0.__class__(last_line_tuple)
        return last_header

    def readable(self):
        """Whether the file can be read and decoded."""
        return self.info.readable

    def _seek_frame(self, index):
        self.fh_ts.seek(self.header0.seek_offset(index))
        if self.header0.mode == 'rawdump':
            self.fh_raw.seek(index * self._payload_nbytes)
        else:
            for fh_pair in self.fh_raw:
                for fh in fh_pair:
                    fh.seek(index * self._payload_nbytes)

    def _fh_raw_read_frame(self):
        return GSBFrame.fromfile(self.fh_ts, self.fh_raw,
                                 payload_nbytes=self._payload_nbytes,
                                 sample_shape=self._unsliced_shape,
                                 bps=self.bps, complex_data=self.complex_data,
                                 verify=self.verify)


class GSBStreamWriter(GSBStreamBase, StreamWriterBase):
    """GSB format writer.

    Encodes and writes sequences of samples to file.

    Parameters
    ----------
    fh_ts : filehandle
        For writing time stamps to storage.
    fh_raw : filehandle, or nested tuple of filehandles
        For writing raw binary data to storage.  A single file is needed for
        rawdump, and a tuple for phased.  For a nested tuple, the outer
        tuple determines the number of polarizations, and the inner tuple(s)
        the number of streams per polarization.  E.g., ``((polL1, polL2),
        (polR1, polR2))`` for two streams per polarization.  A single tuple is
        interpreted as streams of a single polarization.
    header0 : `~baseband.gsb.GSBHeader`
        Header for the first frame, holding time information, etc.
    sample_rate : `~astropy.units.Quantity`, optional
        Number of complete samples per second, i.e. the rate at which each
        channel of each polarization is sampled.  If not given, will be
        inferred assuming the frame rate is exactly 0.25165824 s.
    samples_per_frame : int, optional
        Number of complete samples per frame (possibly combining two files).
        Can give ``payload_nbytes`` instead.
    payload_nbytes : int, optional
        Number of bytes per payload (in each raw file separately).
        If both ``samples_per_frame`` and ``payload_nbytes`` are `None`,
        ``payload_nbytes`` is set to ``2**22`` (4 MiB).
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
    """

    def __init__(self, fh_ts, fh_raw, header0=None, sample_rate=None,
                 samples_per_frame=None, payload_nbytes=None, nchan=None,
                 bps=None, complex_data=None, squeeze=True):
        fh_ts = GSBTimeStampIO(fh_ts)
        super().__init__(
            fh_ts, fh_raw, header0, sample_rate=sample_rate,
            samples_per_frame=samples_per_frame, payload_nbytes=payload_nbytes,
            nchan=nchan, bps=bps, complex_data=complex_data, squeeze=squeeze)
        self._frame = GSBFrame.fromdata(
            np.zeros((self.samples_per_frame,) + self._unsliced_shape,
                     (np.complex64 if self.complex_data else np.float32)),
            header=self.header0.copy(), bps=self.bps)

    def _fh_raw_write_frame(self, frame):
        assert frame.valid
        frame.tofile(self.fh_ts, self.fh_raw)

    def flush(self):
        self.fh_ts.flush()
        try:
            self.fh_raw.flush()
        except AttributeError:
            for fh_pair in self.fh_raw:
                for fh in fh_pair:
                    fh.flush()


class GSBFileOpener(FileOpener):

    non_header_keys = FileOpener.non_header_keys | {'raw'}

    # TODO: think whether the scheme with using FileReader can be made to work.
    def __call__(self, name, mode='rs', **kwargs):
        mode = self.normalize_mode(mode)
        # For binary or timestamp files, the normal opener works fine.
        if mode[1] != 's':
            return super().__call__(name, mode, **kwargs)

        # But for stream mode, we need to open both raw and timestamp.
        fh = self.get_fh(name, mode[0]+'t')
        raw = kwargs.pop('raw', None)
        if raw is None:
            raise TypeError("stream missing required argument 'raw'.")

        stream_mode = kwargs.pop('header_mode',
                                 'phased' if isinstance(raw, (list, tuple))
                                 else 'rawdump')

        if stream_mode == 'rawdump':
            fh_raw = self.get_fh(raw, mode[0]+'b')

        else:
            if not isinstance(raw, (list, tuple)):
                raw = ((raw,),)
            elif not isinstance(raw[0], (list, tuple)):
                raw = (raw,)

            fh_raw = tuple(tuple(self.get_fh(p, mode[0]+'b') for p in pol)
                           for pol in raw)

        if mode == 'ws' and 'header0' not in kwargs:
            kwargs['mode'] = stream_mode
            kwargs['header0'] = self.get_header0(kwargs)

        try:
            return self.classes[mode](fh, fh_raw=fh_raw, **kwargs)
        except Exception:
            if fh is not name:
                fh.close()
            if isinstance(raw, (list, tuple)):
                for pol, polfh in zip(raw, fh_raw):
                    for p, pfh in zip(pol, polfh):
                        if pfh is not p:
                            pfh.close()
            elif fh_raw is not raw:
                fh_raw.close()
            raise


open = GSBFileOpener('GSB', header_class=GSBHeader, classes={
    'rt': GSBTimeStampIO,
    'wt': GSBTimeStampIO,
    'rb': GSBFileReader,
    'wb': GSBFileWriter,
    'rs': GSBStreamReader,
    'ws': GSBStreamWriter}).wrapped(module=__name__, doc="""
Open GSB file(s) for reading or writing.

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
    Number of bytes per payload (in each raw file separately).
    If both ``samples_per_frame`` and ``payload_nbytes`` are `None`,
    ``payload_nbytes`` is set to ``2**22`` (4 MiB).
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

Returns
-------
Filehandle
    :class:`~baseband.gsb.base.GSBTimeStampIO` (timestamp), or
    :class:`~baseband.gsb.base.GSBFileReader` or
    :class:`~baseband.gsb.base.GSBFileWriter` (binary), or
    :class:`~baseband.gsb.base.GSBStreamReader` or
    :class:`~baseband.gsb.base.GSBStreamWriter` (stream)
""")


class GSBFileInfo(FileInfo):
    def get_file_info(self, name, **kwargs):
        info = self._get_info(name, 'rt')
        if self.is_ok(info):
            info.used_kwargs = {}
            if 'raw' in kwargs:
                info.missing.pop('raw')
                info.used_kwargs['raw'] = kwargs['raw']

        return info

    def get_stream_info(self, name, file_info, **kwargs):
        used_kwargs = file_info.used_kwargs
        for key in ('sample_rate', 'payload_nbytes', 'samples_per_frame'):
            if key in kwargs:
                used_kwargs[key] = kwargs[key]

        stream_info = self._get_info(name, mode='rs', **used_kwargs)
        if self.is_ok(stream_info):
            stream_info.used_kwargs = used_kwargs

        return stream_info


info = GSBFileInfo.create(globals())
