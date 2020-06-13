# Licensed under the GPLv3 - see LICENSE
"""Routines to obtain information on baseband files."""
from .helpers import sequentialfile as sf
from . import io as baseband_io
from .vlbi_base.base import FileInfo


__all__ = ['file_info', 'open']


def file_info(name, format=None, **kwargs):
    """Get format and other information from a baseband file.

    The keyword arguments will only be used if needed, so if one is unsure
    what format a file is, but knows it was taken recently and has 8 channels,
    one would put in ``ref_time=Time('2015-01-01'), nchan=8``. Alternatively,
    and perhaps easier, one can first call the function without extra arguments
    in which case the result will describe what is missing.

    Parameters
    ----------
    name : str or filehandle, or sequence of str
        Raw file for which to obtain information.  If a sequence of files is
        passed, returns information from the first file (see Notes).
    format : str, tuple of str, optional
        Formats to try.  If not given, try all standard formats.
    **kwargs
        Any arguments that might help to get information.  For instance,
        Mark 4 and Mark 5B do not have complete timestamps, which can be
        addressed by passing in ``ref_time``.  Furthermore, for Mark 5B, it
        is needed to pass in ``nchan``. Arguments are checked for consistency
        with the file even if not used (see notes below).

    Returns
    -------
    info
        The information on the file, an instance of either
        `~baseband.vlbi_base.file_info.VLBIFileReaderInfo` or
        `~baseband.vlbi_base.file_info.VLBIStreamReaderInfo`.
        Can be turned info a `dict` by calling it (i.e., ``info()``).

    Notes
    -----
    All keyword arguments passed in are classified, ending up in one of
    the following (mostly useful if the file could be opened as a stream):

      - ``used_kwargs``: arguments that were needed to open the file.
      - ``consistent_kwargs``: not needed to open the file, but consistent.
      - ``inconsistent_kwargs``: not needed to open the file, and inconsistent.
      - ``irrelevant_kwargs``: provide information irrelevant for opening.
    """

    # Handle lists and tuples of files, which may be passed from open.
    if isinstance(name, (tuple, list, sf.FileNameSequencer)):
        return file_info(name[0], format, **kwargs)
    # If we're looking at one file but multiple formats, cycle through formats.
    if format is None:
        format = tuple(baseband_io.FORMATS)

    if isinstance(format, tuple):
        for format_ in format:
            info = file_info(name, format_, **kwargs)
            if info:
                break

        return info

    module = getattr(baseband_io, format)
    info_getter = FileInfo(module.open, format)
    return info_getter(name, **kwargs)


def open(name, mode='rs', format=None, **kwargs):
    """Open a baseband file (or sequence of files) for reading or writing.

    Opened as a binary file, one gets a wrapped filehandle that adds
    methods to read/write a frame.  Opened as a stream, the handle is
    wrapped further, and reading and writing to the file is done as if
    the file were a stream of samples.

    Parameters
    ----------
    name : str or filehandle, or sequence of str
        File name, filehandle, or sequence of file names.  A sequence may be a
        list or str of ordered filenames, or an instance of
        `~baseband.helpers.sequentialfile.FileNameSequencer`.
    mode : {'rb', 'wb', 'rs', or 'ws'}, optional
        Whether to open for reading or writing, and as a regular binary
        file or as a stream. Default: 'rs', for reading a stream.
    format : str or tuple of str
        The format the file is in. For reading, this can be a tuple of possible
        formats, all of which will be tried in turn. By default, all supported
        formats are tried.
    **kwargs
        Additional arguments needed for opening the file as a stream.
        For most formats, trying without these will raise an exception that
        tells which arguments are needed. Opening will not succeed if any
        arguments are passed in that are inconsistent with the file, or are
        irrelevant for opening the file.
    """
    if 'w' in mode:
        if format is None or isinstance(format, tuple):
            raise ValueError("cannot specify multiple formats for writing.")
    else:
        info = file_info(name, format, **kwargs)
        if not info:
            if format is None:
                format = tuple(baseband_io.FORMATS)
            raise ValueError("file could not be opened as "
                             + ("any of {}".format(format) if
                                isinstance(format, tuple) else str(format)))
        format = info.format

        if getattr(info, 'missing', False) and 's' in mode:
            raise TypeError("file format {} is missing required arguments {}."
                            .format(format, info.missing))

        if info.inconsistent_kwargs:
            raise ValueError('arguments inconsistent with this {} file were '
                             'passed in: {}'
                             .format(format, info.inconsistent_kwargs))

        if info.irrelevant_kwargs:
            raise TypeError('open() got unexpected keyword arguments {}'
                            .format(info.irrelevant_kwargs))

        kwargs = info.used_kwargs

    module = getattr(baseband_io, format)
    return module.open(name, mode=mode, **kwargs)
