# Licensed under the GPLv3 - see LICENSE
import pytest

from ..header import ParserDict, HeaderParserBase, HeaderParser


class ParserDictSetup:
    @staticmethod
    def create_parser(index, default):
        def parser(words):
            return words[index]
        return parser

    @staticmethod
    def get_default(index, default):
        return default

    @staticmethod
    def random_name(index, default):
        return index


class TestParserDict(ParserDictSetup):
    # Rather minimal tests, since tested so much in practice.
    def test_init(self):
        pd = ParserDict(self.create_parser)
        assert pd.name == 'parsers'
        assert 'Lazily evaluated' in pd.__doc__
        sd = ParserDict(self.random_name, name='setters', doc='random')
        assert sd.name == 'setters'
        assert sd.__doc__ == 'random'
        with pytest.raises(ValueError, match='cannot infer name'):
            ParserDict(self.random_name)


class TestHeaderParserBase(ParserDictSetup):
    def setup_class(cls):
        cls.hp = {'0': (0, 'default0'),
                  '1': (1, 'default1')}

        class H(HeaderParserBase):
            parsers = ParserDict(cls.create_parser)
            indices = ParserDict(cls.random_name, name='indices')
            defaults = ParserDict(cls.get_default)

        cls.H = H

    def test_setup(self):
        assert repr(self.H.parsers).startswith('ParserDict')
        assert "'defaults'" in repr(self.H.defaults)

    def test_init(self):
        h = self.H(self.hp)
        words = ['first', 'second']
        assert h.parsers['0'](words) == 'first'
        assert h.defaults['1'] == 'default1'
        assert h.indices['1'] == 1


class TestHeaderParser:
    def setup_class(cls):
        cls.header_parser = HeaderParser(
            (('x0_16_4', (0, 16, 4)),
             ('x0_31_1', (0, 31, 1, False)),
             ('x1_0_32', (1, 0, 32)),
             ('x2_0_64', (2, 0, 64, 1 << 32))))

    def test_header_parser_update(self):
        extra = HeaderParser((('x4_0_32', (4, 0, 32)),))
        new = self.header_parser + extra
        assert len(new.keys()) == 5
        assert len(self.header_parser.keys()) == 4
        new = self.header_parser.copy()
        assert isinstance(new, HeaderParser)
        new.update(extra)
        assert len(new.keys()) == 5
        assert new['x4_0_32'] == (4, 0, 32)
        with pytest.raises(TypeError):
            self.header_parser + {'x4_0_32': (4, 0, 32)}
        with pytest.raises(ValueError):
            self.header_parser.copy().update(('x4_0_32', (4, 0, 32)))

    def test_header_parser_class(self):
        header_parser = self.header_parser.copy()
        words = [0x12345678, 0xffff0000, 0x0, 0xffffffff]
        header_parser['0_2_8'] = (0, 2, 8, 5)
        assert '0_2_8' in header_parser
        assert header_parser.defaults['0_2_8'] == 5
        assert header_parser.parsers['0_2_8'](words) == (words[0] >> 2) & 0xff
        # Check we can change and parsers will be reset.
        header_parser['0_2_8'] = (0, 1, 8, 3)
        assert '0_2_8' in header_parser
        assert header_parser.defaults['0_2_8'] == 3
        assert header_parser.parsers['0_2_8'](words) == (words[0] >> 1) & 0xff
        header_parser.update({'0_2_8': (0, 3, 8, 1)})
        assert '0_2_8' in header_parser
        assert header_parser.defaults['0_2_8'] == 1
        assert header_parser.parsers['0_2_8'](words) == (words[0] >> 3) & 0xff

        small_parser = HeaderParser((('0_2_8', (0, 2, 8, 4)),))
        header_parser2 = self.header_parser + small_parser
        assert header_parser2.parsers['0_2_8'](words) == (words[0] >> 2) & 0xff
        assert header_parser2.defaults['0_2_8'] == 4
        with pytest.raises(TypeError):
            header_parser + {'0_2_8': (0, 2, 8, 4)}
        with pytest.raises(Exception):
            self.HeaderParser((('0_2_32', (0, 2, 32, 4)),))
        with pytest.raises(Exception):
            self.HeaderParser((('0_2_64', (0, 2, 64, 4)),))
