# Licensed under the GPLv3 - see LICENSE
import sys
from importlib import reload

if sys.version_info >= (3, 8):
    from importlib.metadata import EntryPoint, entry_points
else:
    from importlib_metadata import EntryPoint, entry_points

import pytest

from .. import io as bio, vdif, base

try:
    tasks_entry_points = entry_points(group="baseband.tasks")
except TypeError:
    tasks_entry_points = entry_points().get("baseband.tasks", [])


class TestExistingIOFormat:
    def setup_method(self):
        dir(bio)  # Ensure entries are loaded.
        self.vdif_entry = bio._entries['vdif']

    def teardown_method(self):
        bio._entries['vdif'] = self.vdif_entry
        bio._bad_entries.discard('vdif')
        dir(bio)  # does update.

    def test_io_entry_point(self):
        assert hasattr(bio, 'vdif')
        assert 'vdif' in bio._entries
        assert 'vdif' in bio.FORMATS
        del bio.vdif
        assert 'vdif' not in bio.__dict__
        assert 'vdif' in bio.FORMATS
        assert 'vdif' in dir(bio)
        assert hasattr(bio, 'vdif')
        assert 'vdif' in bio.__dict__
        assert 'vdif' in bio.FORMATS

    def test_fake_bad_vdif(self):
        assert bio.vdif is vdif
        del bio.vdif
        bio._entries['vdif'] = EntryPoint('vdif', 'bad.vdif', '')
        with pytest.raises(AttributeError, match='not loadable'):
            bio.vdif
        assert 'vdif' not in dir(bio)
        assert 'vdif' in bio._bad_entries
        # Does not auto-reload since already known as bad.
        with pytest.raises(AttributeError, match='has no attribute'):
            bio.vdif
        # But will reload if we reload and thus start over.
        reload(bio)
        assert bio.vdif is vdif
        assert 'vdif' in bio.FORMATS


class TestNewIOFormats:
    def setup_entry(self, entry):
        self.added = entry.name
        bio._entries[entry.name] = entry
        bio.FORMATS.append(entry.name)

    def teardown_method(self):
        bio._entries.pop(self.added, None)
        if self.added in bio.FORMATS:
            bio.FORMATS.remove(self.added)
        bio._bad_entries.discard(self.added)
        bio.__dict__.pop(self.added, None)

    def test_find_new(self):
        self.setup_entry(EntryPoint('new', 'baseband.vdif', ''))
        assert 'new' in dir(bio)
        assert 'new' in bio.FORMATS
        assert bio.new is vdif
        # Check that it comes back if we remove it from the module.
        bio.__dict__.pop('new', None)
        assert 'new' not in bio.__dict__
        assert 'new' in bio.FORMATS
        assert 'new' in dir(bio)
        assert bio.new is vdif

    def test_cannot_getattr_bad(self):
        self.setup_entry(EntryPoint('bad', 'really_bad', ''))
        assert 'bad' in dir(bio)
        assert 'bad' in bio.FORMATS
        with pytest.raises(AttributeError, match='not loadable'):
            bio.bad

        assert 'bad' not in dir(bio)
        assert 'bad' not in bio.FORMATS
        with pytest.raises(AttributeError, match='has no attribute'):
            bio.bad

    def test_not_hasattr_bad(self):
        self.setup_entry(EntryPoint('bad', 'really_bad', ''))
        assert 'bad' in dir(bio)
        assert not hasattr(bio, 'bad')
        assert 'bad' not in dir(bio)


class TestTasks:
    def setup_method(self):
        """Remove baseband.tasks so we can reload and find test entry point."""
        # Also ensures that stuff we add gets removed.
        sys.modules.pop('baseband.tasks', None)

    def test_first(self):
        import baseband.tasks as tasks
        assert 'vdif_payload_module' not in dir(tasks)

    def test_task_discovery(self, tmpdir, monkeypatch):
        with open(tmpdir.mkdir('task_tests-0.1.dist-info')
                  .join('entry_points.txt'), 'wt') as fw:
            fw.write('[baseband.tasks]\n'
                     'vdif_payload_module = baseband.vdif.payload\n'
                     'vdif_header_all = baseband.vdif.header:__all__\n'
                     '_nomod = baseband.base.utils:__all__\n')
        monkeypatch.syspath_prepend(str(tmpdir))
        # Now (re-)import tasks
        import baseband.tasks as tasks
        # We loaded just the vdif module.
        assert 'vdif_payload_module' in dir(tasks)
        assert 'vdif' not in dir(tasks)
        assert 'payload' not in dir(tasks)
        assert tasks.vdif_payload_module is vdif.payload
        assert 'VDIFPayload' not in dir(tasks)
        # But helpers and everything in it.
        assert 'vdif_header_all' in dir(tasks)
        assert 'header' not in dir(tasks)
        assert 'VDIFHeader' in dir(tasks)
        assert tasks.vdif_header_all is vdif.header
        assert tasks.VDIFHeader is vdif.header.VDIFHeader
        # And what's in utils, but not the name.
        assert '_nomod' not in dir(tasks)
        assert 'CRC' in dir(tasks)
        assert tasks.CRC is base.utils.CRC

    def test_bad_task_definition(self, tmpdir, monkeypatch):
        with open(tmpdir.mkdir('bad_task_tests-0.1.dist-info')
                  .join('entry_points.txt'), 'wt') as fw:
            fw.write('[baseband.tasks]\n'
                     'vdif_payload_module = baseband.vdif.payload\n'
                     'utils = baseband.base.utils.__all__\n'  # typo: . not :
                     'does_not_exist = baseband.does_not_exist\n')

        monkeypatch.syspath_prepend(str(tmpdir))
        import baseband.tasks as tasks
        assert tasks.vdif_payload_module is vdif.payload
        assert not hasattr(tasks, 'utils')
        assert 'utils' not in dir(tasks)
        assert 'does_not_exist' not in dir(tasks)
        assert len(tasks._bad_entries) == 2
        assert (set(entry.name for entry in tasks._bad_entries)
                == {'utils', 'does_not_exist'})

    @pytest.mark.xfail(tasks_entry_points,
                       reason='cannot test for lack of entry points')
    def test_message_on_empty_tasks(self):
        import baseband.tasks as tasks
        with pytest.raises(AttributeError, match='No.*entry points found'):
            tasks.does_not_exist

    def test_last(self):
        import baseband.tasks as tasks
        assert 'vdif_payload_module' not in dir(tasks)
