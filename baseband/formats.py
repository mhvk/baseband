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
    if 'missing' in info:
        extra_args = {key: kwargs[key] for key in kwargs
                      if key in info['missing']}
        if not extra_args:
            return info

        with module.open(name, mode=mode, **extra_args) as fh:
            info = fh.info()

        info['used_kwargs'] = extra_args

    return info
