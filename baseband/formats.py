# Licensed under the GPLv3 - see LICENSE
"""Routines to obtain information on baseband files."""
import importlib


FILE_FORMATS = {'dada': 'DADA',
                'mark4': 'Mark4',
                'mark5b': 'Mark5B',
                'vdif': 'VDIF',
                'gsb': 'GSB'}


def file_info(name, fmt=None, **kwargs):
    if fmt is None:
        fmt = tuple(FILE_FORMATS.keys())

    if isinstance(fmt, tuple):
        for fmt_ in fmt:
            info = file_info(name, fmt_, **kwargs)
            if info:
                break

        return info

    module = importlib.import_module('.' + fmt, package='baseband')
    # Opening as a binary file (text for GSB) should always work, and allows
    # us to determine whether the file is of the correct format.
    mode = 'rb' if fmt != 'gsb' else 'rt'
    with module.open(name, mode=mode) as fh:
        info = fh.info()

    # If not the right format, return immediately.
    if not info:
        return info

    # If arguments were missing, see if they were passed in.
    extra_args = {}
    if 'missing' in info:
        for key in info['missing']:
            if key in kwargs:
                extra_args[key] = kwargs[key]

        if not extra_args:
            return info

        with module.open(name, mode=mode, **extra_args) as fh:
            info = fh.info()

    # Now see if we should be able to use the stream opener to get
    # even more information.  This is always possible if we have a
    # frame rate, or if a sample_rate was passed on.
    if 'frame_rate' not in info:
        if 'sample_rate' not in kwargs:
            return info

        extra_args['sample_rate'] = kwargs['sample_rate']

    with module.open(name, mode='rs', **extra_args) as fh:
        return fh.info()
