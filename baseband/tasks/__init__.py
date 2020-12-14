# Licensed under the GPLv3 - see LICENSE
"""Analysis tasks.

The tasks are imported via plugins discovered via entry point
'baseband.tasks', such as are provided by the baseband-tasks_
package.  A special rule for the entry points is that if it points
to ``__all__``, all items from that list will be imported.
Furthermore, if any name starts with '_', it is not imported.

Sample entry point entries::

    # Import just a module
    dispersion = baseband_tasks.dispersion
    # Import just one class
    Dedisperse = baseband_tasks.dispersion:Dedisperse
    # Import all entries as well as the module
    dispersion = baseband_tasks.dispersion:__all__
    # Import all entries, but not the module
    _ = baseband_tasks.dispersion:__all__

"""


def _get_entry_points():
    """Get baseband.tasks entry point.

    A list of entries where loading raised an exception are store inside
    the result under '_bad_entries'.
    """
    from importlib import import_module
    import sys
    if sys.version_info >= (3, 8):
        from importlib.metadata import entry_points
    else:
        from importlib_metadata import entry_points

    entries = {'_bad_entries': []}

    for entry_point in entry_points().get('baseband.tasks', []):
        # Only on python >= 3.9 do .module and .attr exist.
        ep_module, _, ep_attr = entry_point.value.partition(':')
        try:
            loaded = entry_point.load()
            if ep_attr == '__all__':
                module = import_module(ep_module)
                entries.update({name: getattr(module, name) for name in loaded
                                if not name.startswith('_')})
                # Possibly load module too, depending on entry point name.
                loaded = module

            if not entry_point.name.startswith('_'):
                entries[entry_point.name] = loaded
        except Exception:
            entries['_bad_entries'].append(entry_point)

    return entries


def __getattr__(attr):
    msg = f"module {__name__!r} has no attribute {attr!r}."
    if not attr.startswith('_'):
        entries = _get_entry_points()
        globals().update(entries)
        if attr in globals():
            return globals()[attr]

        if set(entries) == {'_bad_entries'}:
            msg += ('\nNo baseband.tasks entry points found. '
                    'Maybe baseband-tasks is not installed?')

    raise AttributeError(msg)


def __dir__():
    globals().update(_get_entry_points())
    return sorted(globals())
