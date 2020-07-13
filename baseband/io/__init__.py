# Licensed under the GPLv3 - see LICENSE
"""Entry point for modules for specific file formats.

This will contain the baseband modules as well as any other plugins
discovered via entry point 'baseband.io'.
"""
import entrypoints


__all__ = ['FORMATS', 'BAD_FORMATS']


# Perhaps a bit silly, but it is nice to be able to use a pure source
# checkout even though the baseband entry points are then missing.
# These will normally be overwritten immediately below.
FORMATS = {fmt: entrypoints.EntryPoint(fmt, 'baseband.'+fmt, '')
           for fmt in ('dada', 'guppi', 'mark4', 'mark5b', 'vdif', 'gsb')}
"""Entrypoints to the various formats, keyed by their name."""

BAD_FORMATS = set()
"""Formats for which the entry point failed to load."""

FORMATS.update(entrypoints.get_group_named('baseband.io'))


def __getattr__(fmt):
    entry_point = FORMATS.get(fmt, None)
    if entry_point is None:
        if fmt not in BAD_FORMATS:
            # Try getting entry points again, just in case.
            FORMATS.update(entrypoints.get_group_named('baseband.io'))
            entry_point = FORMATS.get(fmt, None)

        if entry_point is None:
            raise AttributeError(f"baseband.io has no format {fmt!r}")

    try:
        module = entry_point.load()
    except Exception as exc:
        exc.args += (f"{entry_point} was not loadable. Now removed",)
        FORMATS.pop(fmt)
        BAD_FORMATS.add(fmt)
        raise

    # Update so we do not have to go through __getattr__ again.
    globals()[fmt] = module
    return module


def __dir__():
    result = list(globals())
    result.extend(fmt for fmt in FORMATS if fmt not in globals())
    return result
