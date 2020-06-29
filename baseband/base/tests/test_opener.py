# Licensed under the GPLv3 - see LICENSE
import io

import pytest
import numpy as np
from astropy import units as u

from ...helpers import sequentialfile as sf
from ..header import HeaderParser, VLBIHeaderBase, four_word_struct
from ..base import (FileOpener,
                    FileBase, VLBIFileReaderBase,
                    StreamReaderBase, StreamWriterBase)


class BareHeader(VLBIHeaderBase):
    _struct = four_word_struct
    _header_parser = HeaderParser(
        (('x0_16_4', (0, 16, 4)),
         ('x0_31_1', (0, 31, 1, False)),
         ('x1_0_32', (1, 0, 32)),
         ('x2_0_64', (2, 0, 64, 1))))

    @property
    def payload_nbytes(self):
        return 10

    samples_per_frame = 10
    sample_shape = (2,)
    complex_data = False
    dtype = np.dtype('f4')


class BareFileReader(VLBIFileReaderBase):
    pass


class BareFileWriter(FileBase):
    pass


class BareStreamReader(StreamReaderBase):
    pass


class BareStreamWriter(StreamWriterBase):
    def __init__(self, fh_raw, header0, *,
                 squeeze=True, parrot='alife', **kwargs):
        if parrot == 'dead':
            raise ValueError('parrot is dead')
        super().__init__(fh_raw, header0, squeeze=squeeze, **kwargs)


class TestFileOpener:
    def setup_class(cls):
        cls.classes = {'rb': BareFileReader,
                       'wb': BareFileWriter,
                       'rs': BareStreamReader,
                       'ws': BareStreamWriter}
        cls.header_class = BareHeader
        cls.file_opener = FileOpener('Bare', classes=cls.classes,
                                     header_class=cls.header_class)
        cls.open = staticmethod(FileOpener.create(globals(), doc='extra'))

    def test_create_opener(self):
        assert self.open.__wrapped__.__func__ is FileOpener.__call__
        assert 'Open Bare file(s) for reading or writing.' in self.open.__doc__
        assert self.open.__doc__.endswith('extra')

    def test_create_opener_wrong_ns(self):
        with pytest.raises(ValueError, match='does not contain'):
            FileOpener.create(locals(), doc='extra')

    def test_methods(self):
        assert self.file_opener.is_name('abc')
        assert not self.file_opener.is_name(['a', 'b'])
        assert self.file_opener.is_template('{abc}')
        assert self.file_opener.is_sequence('{abc}')
        assert not self.file_opener.is_template('abc')

    def test_get_type_custom_indexer(self):
        class MySeq:
            def __getitem__(self, index):
                return f'{index:07d}.bare'

        my_seq = MySeq()
        assert my_seq[10] == '0000010.bare'
        assert self.file_opener.get_type(my_seq) == 'sequence'
        assert not self.file_opener.is_name(my_seq)
        assert self.file_opener.is_sequence(my_seq)
        assert not self.file_opener.is_template(my_seq)

    @pytest.mark.parametrize('kwargs,expected,left', [
        (dict(), BareHeader.fromvalues(), dict()),
        (dict(x0_16_4=0), BareHeader.fromvalues(x0_16_4=0), dict()),
        (dict(squeeze=True), BareHeader.fromvalues(), dict(squeeze=True)),
        (dict(parrot='dead'), BareHeader.fromvalues(), dict(parrot='dead'))])
    def test_get_header0(self, kwargs, expected, left):
        header0 = self.file_opener.get_header0(kwargs)
        assert header0 == expected
        assert kwargs == left

    def test_get_header0_failure(self):
        with pytest.raises(AttributeError):
            self.file_opener.get_header0(dict(payload_nbytes=2))

    def test_get_fns_with_header_failure(self):
        # But still should be able to get a fns.
        fns = self.file_opener.get_fns('{file_nr}_{bla}.bare', 'rs',
                                       dict(payload_nbytes=2, bla='aha'))
        assert fns[1] == '1_aha.bare'

    def test_binary_name(self, tmpdir):
        name = str(tmpdir.join('test.bare'))
        with self.file_opener(name, 'wb') as fw:
            assert isinstance(fw, BareFileWriter)
            assert fw.fh_raw.name == name
            fw.write(b'abcde')

        with self.file_opener(name, 'rb') as fr:
            assert isinstance(fr, BareFileReader)
            assert fr.fh_raw.name == name
            assert fr.read() == b'abcde'

    def test_binary_fh(self, tmpdir):
        # Also flip mode, and use constructed open
        name = str(tmpdir.join('test.bare'))
        with io.open(name, 'wb') as fh:
            with self.open(fh, 'bw') as fw:
                assert isinstance(fw, BareFileWriter)
                assert fw.fh_raw is fh
                fw.write(b'abcde')

        with io.open(name, 'rb') as fh:
            with self.open(fh, 'br') as fr:
                assert isinstance(fr, BareFileReader)
                assert fr.fh_raw is fh
                assert fr.read() == b'abcde'

    def test_stream_template(self, tmpdir):
        template = str(tmpdir.join('{x2_0_64:03d}_{file_nr:02d}.bare'))
        header0 = BareHeader.fromvalues()
        with self.open(template, 'ws', header0=header0,
                       sample_rate=10*u.Hz) as fw:
            assert isinstance(fw, BareStreamWriter)
            assert isinstance(fw.fh_raw, sf.SequentialFileWriter)
            assert isinstance(fw.fh_raw.files, sf.FileNameSequencer)
            assert fw.fh_raw.files[0].endswith('001_00.bare')

        with self.open(template, 'w', x2_0_64=4,
                       sample_rate=10*u.Hz, parrot='life') as fw:
            assert fw.fh_raw.files[0].endswith('004_00.bare')

        with pytest.raises(TypeError):
            self.open(template, 'ws', coocoo=10,
                      sample_rate=10*u.Hz)

    @pytest.mark.parametrize('module,doc', [
        (None, None),
        (None, 'new_doc'),
        (__name__, 'new_doc')])
    def test_wrapped(self, module, doc):
        open = self.file_opener.wrapped(module, doc)
        if doc is None:
            assert open.__doc__ is self.file_opener.__call__.__doc__
        else:
            assert open.__doc__ is doc

        if module is None:
            assert open.__module__ is self.file_opener.__call__.__module__
        else:
            assert open.__module__ is module

    def test_invalid_name(self):
        with pytest.raises(ValueError, match='not understood'):
            self.file_opener(0, 'wb')

    def test_invalid_mode(self):
        name = 'not_needed'
        with pytest.raises(ValueError, match='invalid mode'):
            self.file_opener(name, 'xb')

        with pytest.raises(ValueError, match='invalid mode'):
            self.file_opener(name, 's')

    def test_binary_no_template(self):
        with pytest.raises(ValueError, match='does not support'):
            self.file_opener('{file_nr}.bare', 'wb')
