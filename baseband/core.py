# Licensed under the GPLv3 - see LICENSE
"""Routines to obtain information on baseband files."""
import importlib


FILE_FORMATS = {'dada': 'DADA',
                'mark4': 'Mark4',
                'mark5b': 'Mark5B',
                'vdif': 'VDIF',
                'gsb': 'GSB'}


def file_info(name, format=None, **kwargs):
    """Get format and other information from a baseband file.

    Parameters
    ----------
    name : str or filehandle
        Raw file for which to obtain information.
    format : str, tuple of str, optional
        Formats to try.  If not given, try all standard formats.
    **kwargs
        Any arguments that might help to get information.  For instance,
        Mark 4 and Mark 5B do not have complete timestamps, which can be
        addressed by passing in ``ref_time``.  Furthermore, for Mark 5B, it
        is needed to pass in ``nchan``.

    Returns
    -------
    info : `~baseband.vlbi_base.file_info.VLBIFileReaderInfo` or `~baseband.vlbi_base.file_info.VLBIStreamReaderInfo`
        The information on the file. Can be turned info a `dict` by calling it
        (i.e., ``info()``).  Any extra keywords used in opening the file are
        attached as an ``kwargs`` attribute.

    Notes
    -----
    The keyword arguments will only be used if needed, so if one is unsure
    what format a file is, but knows it was taken recently and has 8 channels,
    one would put in ``ref_time=Time('2015-01-01'), nchan=8``. Alternatively,
    and perhaps easier, one can first call the function without extra arguments
    in which case the result will describe what is missing.
    """
    if format is None:
        format = tuple(FILE_FORMATS.keys())

    if isinstance(format, tuple):
        for format_ in format:
            info = file_info(name, format_, **kwargs)
            if info:
                break

        return info

    module = importlib.import_module('.' + format, package='baseband')
    # Opening as a binary file (text for GSB) should always work, and allows
    # us to determine whether the file is of the correct format.
    mode = 'rb' if format != 'gsb' else 'rt'
    with module.open(name, mode=mode) as fh:
        info = fh.info

    # If not the right format, return immediately.
    if not info:
        return info

    # If arguments were missing, see if they were passed in.
    if info.missing:
        extra_args = {key: kwargs[key] for key in info.missing
                      if key in kwargs}

        if not extra_args:
            return info

        if format != 'gsb':
            with module.open(name, mode=mode, **extra_args) as fh:
                info = fh.info

            if info.missing:
                return info
    else:
        extra_args = {}

    # Now see if we should be able to use the stream opener to get
    # even more information.  This is always possible if we have a
    # frame rate, or if a sample_rate was passed on.
    if info.frame_rate is None:
        if 'sample_rate' not in kwargs:
            return info

        extra_args['sample_rate'] = kwargs['sample_rate']

    with module.open(name, mode='rs', **extra_args) as fh:
        info = fh.info

    info.kwargs = extra_args
    return info
