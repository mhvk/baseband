# Licensed under the GPLv3 - see LICENSE
"""Baseband general I/O routines and entry point.

Contains general ``open`` and ``file_info`` functions that can iterate
over possible formats to determine which is the right one, including
possible formats discovered via entry point 'baseband.io', such as
the `~baseband_tasks.io.hdf5` module from the baseband-tasks_ package.

Any 'baseband.io' entry points are treated as possible formats if they
point to a module (e.g., 'vdif = baseband.vdif').  Any entries that
have object names (e.g., 'fancy_open = mypackage.io:open') are just
added to the name space.

Attributes
----------
FORMATS : list
    Available baseband formats.

"""
import sys

import entrypoints


__all__ = ['open', 'file_info']


__self__ = sys.modules[__name__]
"""Link to our own module, for convenience below."""

# We only load entries on demand, to keep import time minimal.
_entries = {}
"""Entry points found."""
_bad_entries = set()
"""Any entry points that failed to load. These will not be retried."""


def __getattr__(attr):
    """Get a missing attribute from a possible entry point.

    Looks for the attribute among the (possibly updated) entry points,
    and, if found, tries loading the entry.  If that fails, the entry
    is added to _bad_entries to ensure it does not recur.
    """
    if attr.startswith('_') or attr in _bad_entries:
        raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")

    FORMATS = globals().setdefault('FORMATS', [])
    if attr not in _entries:
        if not _entries:
            # On initial update, we add our own formats as explicit entries,
            # in part to set some order, but also so things work even in a
            # pure source checkout, where entry points are missing.
            _entries.update({
                fmt: entrypoints.EntryPoint(fmt, 'baseband.'+fmt, '')
                for fmt in ('dada', 'guppi', 'mark4', 'mark5b', 'vdif', 'gsb')
            })

        _entries.update(entrypoints.get_group_named('baseband.io'))
        FORMATS.extend([name for name, entry in _entries.items()
                        if not (entry.object_name or name in FORMATS)])
        if attr == 'FORMATS':
            return FORMATS

    entry = _entries.get(attr, None)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")

    try:
        value = entry.load()
    except Exception:
        _entries.pop(attr)
        _bad_entries.add(attr)
        if attr in FORMATS:
            FORMATS.remove(attr)
        raise AttributeError(f"{entry} was not loadable. Now removed")

    # Update so we do not have to go through __getattr__ again.
    globals()[attr] = value
    return value


def __dir__():
    # Force update of entries, creates 'FORMATS' if it doesn't exist.
    hasattr(__self__, 'absolutely_no_way_this_exists')
    return sorted(set(globals()).union(_entries).difference(_bad_entries))


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
        The information on the file.  Depending on how much information could
        be gathered, this will be an instance of either
        `~baseband.base.file_info.StreamReaderInfo`,
        `~baseband.base.file_info.FileReaderInfo`, or
        `~baseband.base.file_info.NoInfo`.

    Notes
    -----
    All keyword arguments passed in are classified, ending up in one of
    the following (mostly useful if the file could be opened as a stream):

      - ``used_kwargs``: arguments that were needed to open the file.
      - ``consistent_kwargs``: not needed to open the file, but consistent.
      - ``inconsistent_kwargs``: not needed to open the file, and inconsistent.
      - ``irrelevant_kwargs``: provide information irrelevant for opening.
    """
    from baseband.base.file_info import NoInfo

    # If we're looking at one file but multiple formats, cycle through formats.
    if format is None:
        format = tuple(__self__.FORMATS)

    if isinstance(format, tuple):
        no_info = set()
        for format_ in format:
            info = file_info(name, format_, **kwargs)
            if info:
                return info

            if isinstance(info, NoInfo):
                no_info.add(format_)

        return NoInfo(f"{name} does not seem formatted as any of {set(format)}"
                      + (f" ({no_info} had no 'info' and opening failed)."
                         if no_info else "."))

    module = getattr(__self__, format)
    # A well-behaved module should define info, but we allow for more
    # minimally defined ones by trying to open the file and getting info
    # from it (which may well work if VLBIStreamReaderBase is subclasses).
    if hasattr(module, 'info'):
        return module.info(name, **kwargs)

    # Try just opening as stream and getting info.
    try:
        with module.open(name, 'rs', **kwargs) as fh:
            return fh.info
    except Exception as exc:
        return NoInfo(f"baseband.io.{format} has no 'info' and opening "
                      f"raised {exc!r}.")


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
        The format the file is in. For reading, if a tuple of possible formats,
        all will be tried in turn. By default, all supported formats are tried.
        For writing, an explicit format must be passed in.
    **kwargs
        Additional arguments needed for opening the file as a stream.
        For most formats, trying without these will raise an exception that
        tells which arguments are needed. Opening will not succeed if any
        arguments are passed in that are inconsistent with the file, or are
        irrelevant for opening the file.
    """
    if format is None or isinstance(format, tuple):
        if 'w' in mode:
            raise ValueError("cannot specify multiple formats for writing.")

        info = file_info(name, format, **kwargs)
        if not info:
            raise ValueError("format of file could not be auto-determined")

        format = info.format

        if getattr(info, 'missing', False) and 's' in mode:
            raise TypeError(f"file format {format} is missing required "
                            f"arguments {info.missing}.")

        if getattr(info, 'inconsistent_kwargs', False):
            raise ValueError(f"arguments inconsistent with this {format} file "
                             f"were passed in: {info.inconsistent_kwargs}")

        if getattr(info, 'irrelevant_kwargs', False):
            raise TypeError(f"open() got unexpected keyword arguments "
                            f"{info.irrelevant_kwargs}")

        kwargs = getattr(info, 'used_kwargs', kwargs)

    module = getattr(__self__, format)
    return module.open(name, mode=mode, **kwargs)
