# Licensed under the GPLv3 - see LICENSE
"""Entry point for modules for specific file formats.

This will contain the baseband modules as well as any other plugins
discovered via entry point 'baseband.io'.

Attributes
----------
FORMATS : list
    Available baseband formats.

"""
import sys

import entrypoints


__all__ = []


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
