# Licensed under the GPLv3 - see LICENSE
import os

import pytest
import entrypoints

from .. import io as bio, vdif


# If we're in a source checkout in which not even setup.py egg_info has
# been run, the entry points cannot be found so we should skip tests.
pure_source_checkout = (
    os.path.exists(os.path.join(os.path.dirname(__file__),
                                '..', '..', 'setup.cfg'))
    and 'vdif' not in entrypoints.get_group_named('baseband.io'))


def test_io_entry_point():
    assert 'vdif' in bio.FORMATS
    assert isinstance(bio.FORMATS['vdif'], entrypoints.EntryPoint)
    bio.__dict__.pop('vdif', None)
    assert 'vdif' not in bio.__dict__
    assert 'vdif' in dir(bio)
    assert hasattr(bio, 'vdif')
    assert 'vdif' in bio.__dict__
    assert bio.vdif is vdif
    assert 'vdif' in bio.FORMATS


class TestBadFormat:
    def setup_method(self):
        bio.FORMATS['bad'] = entrypoints.EntryPoint('bad', 'really_bad', '')

    def teardown_method(self):
        bio.BAD_FORMATS.discard('bad')
        bio.FORMATS.pop('bad', None)

    def test_cannot_getattr_bad(self):
        assert 'bad' in dir(bio)
        assert 'bad' not in bio.BAD_FORMATS
        with pytest.raises(AttributeError, match='not loadable'):
            bio.bad

        assert 'bad' not in bio.FORMATS
        assert 'bad' in bio.BAD_FORMATS

    def test_not_hasattr_bad(self):
        assert 'bad' in bio.FORMATS
        assert 'bad' not in bio.BAD_FORMATS
        assert not hasattr(bio, 'bad')
        assert 'bad' not in bio.FORMATS
        assert 'bad' in bio.BAD_FORMATS


@pytest.mark.skipif(pure_source_checkout,
                    reason='pure source checkout with undefined entry points')
def test_fake_bad_vdif():
    old_vdif = bio.FORMATS['vdif']
    try:
        del bio.vdif
        assert 'vdif' in dir(bio)
        assert bio.vdif is vdif
        del bio.vdif
        bio.FORMATS['vdif'] = entrypoints.EntryPoint('vdif', 'bad.vdif', '')
        assert 'vdif' in dir(bio)
        with pytest.raises(AttributeError, match='not loadable'):
            bio.vdif
        assert 'vdif' not in dir(bio)
        assert 'vdif' in bio.BAD_FORMATS
        # Does not auto-reload since already known as bad.
        with pytest.raises(AttributeError, match='has no format'):
            bio.vdif
        # But will reload if we discard bad format.
        bio.BAD_FORMATS.discard('vdif')
        assert bio.vdif is vdif
    finally:
        bio.FORMATS['vdif'] = old_vdif
        bio.BAD_FORMATS.discard('vdif')
        bio.__dict__['vdif'] = vdif
