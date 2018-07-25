# Licensed under the GPLv3 - see LICENSE
"""Routines to obtain information on baseband files."""
import importlib
import numpy as np

from .helpers import sequentialfile as sf

__all__ = ['file_info', 'open']

FILE_FORMATS = ('dada', 'mark4', 'mark5b', 'vdif', 'guppi', 'gsb')


def file_info(name, format=FILE_FORMATS, **kwargs):
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
    info : `~baseband.vlbi_base.file_info.VLBIFileReaderInfo` or `~baseband.vlbi_base.file_info.VLBIStreamReaderInfo`
        The information on the file. Can be turned info a `dict` by calling it
        (i.e., ``info()``).

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
    elif isinstance(format, tuple):
        for format_ in format:
            info = file_info(name, format_, **kwargs)
            if info:
                break

        return info

    module = importlib.import_module('.' + format, package='baseband')
    # Opening as a binary file (text for GSB) should always work, and allows
    # us to determine whether the file is of the correct format.  Here, getting
    # info should never fail or even emit warnings (i.e., if tests start to
    # give warnings, info should be fixed, not a filter done here).
    mode = 'rb' if format != 'gsb' else 'rt'
    with module.open(name, mode=mode) as fh:
        info = fh.info

    # If not the right format, return immediately.
    if not info:
        return info

    # If arguments were missing, see if they were passed in.
    if info.missing:
        used_kwargs = {key: kwargs[key] for key in info.missing
                       if key in kwargs}

        if used_kwargs:
            if format == 'gsb':
                # 'raw' keyword not useful for opening the timestamp file.
                # Just remove from info.missing.
                info.missing.pop('raw')
            else:
                with module.open(name, mode=mode, **used_kwargs) as fh:
                    info = fh.info

    else:
        used_kwargs = {}

    if not info.missing:
        # Now see if we should be able to use the stream opener to get
        # even more information.  If there no longer are missing arguments,
        # then this should always be possible if we have a frame rate, or
        # if a sample_rate was passed on.
        frame_rate = info.frame_rate
        if frame_rate is None and 'sample_rate' in kwargs:
            used_kwargs['sample_rate'] = kwargs['sample_rate']
            frame_rate = 'known'

        if frame_rate is not None:
            with module.open(name, mode='rs', **used_kwargs) as fh:
                info = fh.info

    # Store what happened to the kwargs, so one can decide if there are
    # inconsistencies or other problems.
    info.used_kwargs = used_kwargs
    info.consistent_kwargs = {}
    info.inconsistent_kwargs = {}
    info.irrelevant_kwargs = {}
    info_dict = info()
    info_dict.update(info_dict.pop('file_info', {}))
    for key, value in kwargs.items():
        if key in used_kwargs:
            continue
        info_value = info_dict.get(key)
        consistent = None
        if info_value is not None:
            consistent = info_value == value

        elif key == 'nchan':
            sample_shape = info_dict.get('sample_shape')
            if sample_shape is not None:
                # If we passed nchan, and info doesn't have it, but does have a
                # sample shape, check that consistency with that, either in
                # being equal to `sample_shape.nchan` or equal to the product
                # of all elements (e.g., a VDIF file with 8 threads and 1
                # channel per thread is consistent with nchan=8).
                consistent = (getattr(sample_shape, 'nchan', -1) == value or
                              np.prod(sample_shape) == value)

        elif key in {'ref_time', 'kday', 'decade'}:
            start_time = info_dict.get('start_time')
            if start_time is not None:
                if key == 'ref_time':
                    consistent = abs(value - start_time).jd < 500
                elif key == 'kday':
                    consistent = int(start_time.mjd / 1000.) * 1000 == value
                else:  # decade
                    consistent = int(start_time.isot[:3]) * 10 == value

        if consistent is None:
            info.irrelevant_kwargs[key] = value
        elif consistent:
            info.consistent_kwargs[key] = value
        else:
            info.inconsistent_kwargs[key] = value

    return info


def open(name, mode='rs', format=FILE_FORMATS, **kwargs):
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
        if isinstance(format, tuple):
            raise ValueError("cannot specify multiple formats for writing.")
    else:
        info = file_info(name, format, **kwargs)
        if not info:
            raise ValueError("file could not be opened as " +
                             ("any of {}".format(format) if
                              isinstance(format, tuple) else str(format)))
        format = info.format

        if info.missing and 's' in mode:
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

    module = importlib.import_module('.' + format, package='baseband')
    return module.open(name, mode=mode, **kwargs)
