# Licensed under the GPLv3 - see LICENSE
"""Entry point for modules for specific file formats.

Contains the baseband modules as well as any plugins discovered via
entry point 'baseband.io'.
"""


def _get_entry_points():
    import importlib

    class BasebandFormat:
        def __init__(self, name, value):
            self.name = name
            self.value = value

        def load(self):
            return importlib.import_module(self.value)

        def __repr__(self):
            return f"BasebandFormat('{self.name}', '{self.value}')"

    entries = {key: BasebandFormat(key, 'baseband.'+key) for key
               in ('dada', 'guppi', 'mark4', 'mark5b', 'vdif', 'gsb')}
    try:
        from entrypoints import get_group_all
    except ImportError:
        try:
            from pkg_resources import iter_entry_points as get_group_all
        except ImportError:
            return entries

    for entry_point in get_group_all('baseband.io'):
        entries.setdefault(entry_point.name, entry_point)

    return entries


FORMATS = _get_entry_points()


def __getattr__(name):
    try:
        return FORMATS[name].load()
    except KeyError:
        raise AttributeError
    except Exception as exc:
        exc.args += (f'{FORMATS[name]} was not loadable. Now removed',)
        FORMATS.pop(name)
        raise


# Clean up namespace
del _get_entry_points
