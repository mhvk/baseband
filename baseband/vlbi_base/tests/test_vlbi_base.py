# Licensed under the GPLv3 - see LICENSE
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from copy import copy
import io
import numpy as np
import pytest
import astropy.units as u
from astropy.tests.helper import catch_warnings
from collections import namedtuple
from ..utils import bcd_encode, bcd_decode, CRC, gcd, lcm
from ..header import HeaderParser, VLBIHeaderBase, four_word_struct
from ..payload import VLBIPayloadBase
from ..frame import VLBIFrameBase
from ..base import (VLBIFileBase, VLBIStreamBase,
                    VLBIStreamReaderBase, VLBIStreamWriterBase)


def encode_1bit(values):
    return np.packbits(values.ravel())


def decode_1bit(values):
    return np.unpackbits(values.view(np.uint8)).astype(np.float32)


def encode_8bit(values):
    return np.clip(np.round(values),
                   -128, 127).astype(np.int8)


def decode_8bit(values):
    return values.view(np.int8).astype(np.float32)


class Payload(VLBIPayloadBase):
    _encoders = {1: encode_1bit,
                 8: encode_8bit}
    _decoders = {1: decode_1bit,
                 8: decode_8bit}


class TestBCD(object):
    def test_bcd_decode(self):
        assert bcd_decode(0x1) == 1
        assert bcd_decode(0x9123) == 9123
        with pytest.raises(ValueError):
            bcd_decode(0xf)
        decoded = bcd_decode(np.array([0x1, 0x9123]))
        assert isinstance(decoded, np.ndarray)
        assert np.all(decoded == np.array([1, 9123]))
        with pytest.raises(ValueError):
            bcd_decode(np.array([0xf, 9123]))
        with pytest.raises(TypeError):
            bcd_decode([1, 2])

    def test_bcd_encode(self):
        assert bcd_encode(1) == 0x1
        assert bcd_encode(9123) == 0x9123
        with pytest.raises(TypeError):
            bcd_encode('bla')

    def test_roundtrip(self):
        assert bcd_decode(bcd_encode(15)) == 15
        assert bcd_decode(bcd_encode(8765)) == 8765
        a = np.array([1, 9123])
        assert np.all(bcd_decode(bcd_encode(a)) == a)


class TestVLBIBase(object):
    def setup(self):
        self.header_parser = HeaderParser(
            (('x0_16_4', (0, 16, 4)),
             ('x0_31_1', (0, 31, 1, False)),
             ('x1_0_32', (1, 0, 32)),
             ('x2_0_64', (2, 0, 64, 1 << 32))))

        class Header(VLBIHeaderBase):
            _struct = four_word_struct
            _header_parser = self.header_parser
            payload_nbytes = 8

        self.Header = Header
        self.header = self.Header([0x12345678, 0xffff0000, 0x0, 0xffffffff])
        self.Payload = Payload
        self.payload = Payload(np.array([0x12345678, 0xffff0000],
                                        dtype=Payload._dtype_word),
                               sample_shape=(2,), bps=8, complex_data=False)
        self.payload1bit = Payload(np.array([0x12345678]*5,
                                            dtype=Payload._dtype_word),
                                   sample_shape=(5,), bps=1, complex_data=True)

        class Frame(VLBIFrameBase):
            _header_class = Header
            _payload_class = Payload

        self.Frame = Frame
        self.frame = Frame(self.header, self.payload)

    def test_header_parser_update(self):
        extra = HeaderParser((('x4_0_32', (4, 0, 32)),))
        new = self.header_parser + extra
        assert len(new.keys()) == 5
        new = self.header_parser.copy()
        new.update(extra)
        assert len(new.keys()) == 5
        with pytest.raises(TypeError):
            self.header_parser + {'x4_0_32': (4, 0, 32)}
        with pytest.raises(TypeError):
            self.header_parser.copy().update(('x4_0_32', (4, 0, 32)))

    def test_header_basics(self):
        header = self.Header(None)
        assert header.words == [0] * 4
        with pytest.raises(Exception):
            self.Header([1] * 5)
        with pytest.raises(Exception):
            self.Header([1] * 3)
        header = self.header.copy()
        assert header == self.header
        assert header is not self.header
        header = copy(self.header)
        assert header == self.header
        assert header is not self.header
        assert type(header.words) is list
        header.mutable = False
        assert type(header.words) is tuple
        header = self.Header(0, verify=False)
        with pytest.raises(Exception):
            header.verify()
        with pytest.raises(TypeError):
            header.mutable = True

    def test_header_fromfile(self, tmpdir):
        with open(str(tmpdir.join('test.dat')), 'w+b') as s:
            s.write(four_word_struct.pack(*self.header.words))
            s.seek(2)
            with pytest.raises(EOFError):
                self.Header.fromfile(s)
            s.seek(0)
            header = self.Header.fromfile(s)
        assert header == self.header

    def test_parser(self):
        """Test that parsers work as expected."""
        assert self.header['x0_16_4'] == 4
        assert self.header['x0_31_1'] is False
        assert self.header['x1_0_32'] == self.header.words[1]
        assert (self.header['x2_0_64'] ==
                self.header.words[2] + self.header.words[3] * (1 << 32))
        assert 'x0_31_1' in self.header
        assert 'bla' not in self.header
        with pytest.raises(KeyError):
            self.header['bla']
        with pytest.raises(KeyError):
            self.header['bla'] = 1
        with pytest.raises(AttributeError):
            self.header.x0_16_4
        with pytest.raises(AttributeError):
            self.header.xbla

    def test_make_setter(self):
        header = self.header.copy()
        header['x0_16_4'] = 0xf
        assert header.words[0] == 0x123f5678
        with pytest.raises(ValueError):
            header['x0_16_4'] = 0x10
        header['x0_31_1'] = True
        assert header.words[0] == 0x923f5678
        header['x1_0_32'] = 0x1234
        assert header.words[:2] == [0x923f5678, 0x1234]
        header['x2_0_64'] = 1
        assert header.words[2:] == [1, 0]
        header['x2_0_64'] = None
        assert header.words[2:] == [0, 1]
        # Also check update method.
        header.update(x1_0_32=0x5678, x2_0_64=1)
        assert header.words == [0x923f5678, 0x5678, 1, 0]
        with catch_warnings(UserWarning) as w:
            header.update(bla=10)
        assert 'unused' in str(w[0].message)
        assert 'bla' in str(w[0].message)

    def test_header_parser_class(self):
        header_parser = self.header_parser
        words = self.header.words
        header_parser['0_2_8'] = (0, 2, 8, 5)
        assert '0_2_8' in header_parser
        assert header_parser.defaults['0_2_8'] == 5
        assert header_parser.parsers['0_2_8'](words) == (words[0] >> 2) & 0xff
        small_parser = HeaderParser((('0_2_8', (0, 2, 8, 4)),))
        header_parser2 = self.header_parser + small_parser
        assert header_parser2.parsers['0_2_8'](words) == (words[0] >> 2) & 0xff
        assert header_parser2.defaults['0_2_8'] == 4
        with pytest.raises(TypeError):
            header_parser + {'0_2_8': (0, 2, 8, 4)}
        with pytest.raises(TypeError):
            header_parser + {'0_2_8': (0, 2, 8, 4)}
        with pytest.raises(Exception):
            self.HeaderParser((('0_2_32', (0, 2, 32, 4)),))
        with pytest.raises(Exception):
            self.HeaderParser((('0_2_64', (0, 2, 64, 4)),))

    def test_payload_basics(self):
        assert self.payload.complex_data is False
        assert self.payload.sample_shape == (2,)
        assert self.payload.bps == 8
        assert self.payload.nbytes == 8
        assert self.payload.shape == (4, 2)
        assert self.payload.size == 8
        assert self.payload.ndim == 2
        assert np.all(self.payload.data.ravel() ==
                      self.payload.words.view(np.int8))
        assert np.all(np.array(self.payload).ravel() ==
                      self.payload.words.view(np.int8))
        assert np.all(np.array(self.payload, dtype=np.int8).ravel() ==
                      self.payload.words.view(np.int8))
        payload = self.Payload(self.payload.words, bps=4)
        with pytest.raises(KeyError):
            payload.data
        with pytest.raises(ValueError):
            self.Payload(self.payload.words.astype('>u4'), bps=4)
        payload = self.Payload(self.payload.words, bps=8, complex_data=True)
        assert np.all(payload.data ==
                      self.payload.data[:, 0] + 1j * self.payload.data[:, 1])

        assert self.payload1bit.complex_data is True
        assert self.payload1bit.sample_shape == (5,)
        assert self.payload1bit.bps == 1
        assert self.payload1bit.shape == (16, 5)
        assert self.payload1bit.nbytes == 20
        assert np.all(self.payload1bit.data.ravel() ==
                      np.unpackbits(self.payload1bit.words.view(np.uint8))
                      .astype(np.float32).view(np.complex64))

    @pytest.mark.parametrize('item', (2, slice(1, 3), (), slice(2, None),
                                      (2, 1), (slice(None), 0),
                                      (slice(1, 3), 1)))
    def test_payload_getitem_setitem(self, item):
        data = self.payload.data
        sel_data = data[item]
        assert np.all(self.payload[item] == sel_data)
        payload = self.Payload(self.payload.words.copy(), sample_shape=(2,),
                               bps=8, complex_data=False)
        assert payload == self.payload
        payload[item] = 1 - sel_data
        check = self.payload.data
        check[item] = 1 - sel_data
        assert np.all(payload[item] == 1 - sel_data)
        assert np.all(payload.data == check)
        assert np.all(payload[:] ==
                      payload.words.view(np.int8).reshape(-1, 2))
        assert payload != self.payload
        payload[item] = sel_data
        assert np.all(payload[item] == sel_data)
        assert payload == self.payload
        payload = self.Payload.fromdata(data + 1j * data, bps=8)
        sel_data = payload.data[item]
        assert np.all(payload[item] == sel_data)
        payload[item] = 1 - sel_data
        check = payload.data
        check[item] = 1 - sel_data
        assert np.all(payload.data == check)

    def test_payload_bad_fbps(self):
        with pytest.raises(TypeError):
            self.payload1bit[10:11]

    def test_payload_empty_item(self):
        p11 = self.payload[1:1]
        assert p11.size == 0
        assert p11.shape == (0,) + self.payload.sample_shape
        assert p11.dtype == self.payload.dtype
        payload = self.Payload(self.payload.words.copy(), sample_shape=(2,),
                               bps=8, complex_data=False)
        payload[1:1] = 1
        assert payload == self.payload

    @pytest.mark.parametrize('item', (20, -20, (slice(None), 5)))
    def test_payload_invalid_item(self, item):
        with pytest.raises(IndexError):
            self.payload[item]

        payload = self.Payload(self.payload.words.copy(), sample_shape=(2,),
                               bps=8, complex_data=False)
        with pytest.raises(IndexError):
            payload[item] = 1

    def test_payload_invalid_item2(self):
        with pytest.raises(TypeError):
            self.payload['l']
        payload = self.Payload(self.payload.words.copy(), sample_shape=(2,),
                               bps=8, complex_data=False)
        with pytest.raises(TypeError):
            payload['l'] = 1

    def test_payload_setitem_wrong_shape(self):
        payload = self.Payload(self.payload.words.copy(), sample_shape=(2,),
                               bps=8, complex_data=False)
        with pytest.raises(ValueError):
            payload[1] = np.ones(10)

        with pytest.raises(ValueError):
            payload[1] = np.ones((2, 2))

        with pytest.raises(ValueError):
            payload[1:3] = np.ones((2, 3))

        with pytest.raises(ValueError):
            payload[1:3, 0] = np.ones((2, 2))

        with pytest.raises(ValueError):
            payload[1:3, :1] = np.ones((1, 2))

    def test_payload_fromfile(self, tmpdir):
        with open(str(tmpdir.join('test.dat')), 'w+b') as s:
            self.payload.tofile(s)
            s.seek(0)
            with pytest.raises(ValueError):
                self.Payload.fromfile(s)  # No size given.
            s.seek(0)
            payload = self.Payload.fromfile(
                s, payload_nbytes=len(self.payload.words) * 4,
                sample_shape=(2,), bps=8)
        assert payload == self.payload

    def test_payload_fromdata(self):
        data = np.random.normal(0., 64., 16).reshape(16, 1)
        payload = self.Payload.fromdata(data, bps=8)
        assert payload.complex_data is False
        assert payload.sample_shape == (1,)
        assert payload.bps == 8
        assert payload.words.dtype is self.Payload._dtype_word
        assert len(payload.words) == 4
        assert len(payload) == len(data)
        assert payload.nbytes == 16
        payload2 = self.Payload.fromdata(self.payload.data,
                                         bps=self.payload.bps)
        assert payload2 == self.payload
        header = self.header.copy()
        header.bps = 8
        payload3 = self.Payload.fromdata(self.payload.data,
                                         header=header)
        assert payload3 == self.payload
        payload4 = self.Payload.fromdata(data.ravel(), bps=8)
        assert payload4.sample_shape == ()
        assert payload4.shape == (16,)
        assert payload4 != payload
        assert np.all(payload4.data == payload.data.ravel())
        with pytest.raises(ValueError):  # don't have relevant encoder.
            self.Payload.fromdata(data, bps=4)
        payload5 = self.Payload.fromdata(data[::2, 0] + 1j * data[1::2, 0],
                                         bps=8)
        assert payload5.complex_data is True
        assert payload5.sample_shape == ()
        assert payload5.shape == (8,)
        assert payload5 != payload
        assert np.all(payload5.words == payload.words)

    def test_frame_basics(self):
        assert self.frame.header is self.header
        assert self.frame.payload is self.payload
        assert len(self.frame) == len(self.payload)
        assert self.frame.sample_shape == self.payload.sample_shape
        assert self.frame.shape == self.payload.shape
        assert self.frame.size == self.payload.size
        assert self.frame.ndim == self.payload.ndim
        assert np.all(self.frame.data == self.payload.data)
        assert np.all(np.array(self.frame) == np.array(self.payload))
        assert np.all(np.array(self.frame, dtype=np.float64) ==
                      np.array(self.payload))
        assert self.frame.valid is True
        frame = self.Frame(self.header, self.payload, valid=False)
        assert np.all(frame.data == 0.)
        frame.fill_value = 1.
        assert np.all(frame.data == 1.)

        assert 'x2_0_64' in self.frame
        assert self.frame['x2_0_64'] == self.header['x2_0_64']
        for item in (3, (1, 1), slice(0, 2)):
            assert np.all(self.frame[item] == self.frame.payload[item])

        frame2 = self.Frame(self.header.copy(),
                            self.Payload.fromdata(self.payload.data,
                                                  bps=self.payload.bps))

        assert frame2.header == self.header
        assert frame2.payload == self.payload
        assert frame2 == self.frame
        frame2['x2_0_64'] = 0x1
        assert frame2['x2_0_64'] == 0x1
        assert frame2.header != self.header
        frame2[3, 1] = 5.
        assert frame2[3, 1] == 5
        assert frame2.payload != self.payload

    def test_frame_fromfile(self, tmpdir):
        with open(str(tmpdir.join('test.dat')), 'w+b') as s:
            self.frame.tofile(s)
            s.seek(0)
            frame = self.Frame.fromfile(s, payload_nbytes=self.payload.nbytes,
                                        sample_shape=(2,), bps=8)
        assert frame == self.frame

    def test_frame_fromdata(self):
        frame = self.Frame.fromdata(self.frame.data, self.header, bps=8)
        assert frame == self.frame
        frame2 = self.Frame.fromdata(self.frame.data, self.header,
                                     bps=8, valid=False)
        assert np.all(frame2.data == 0.)

    def test_vlbi_file_base(self, tmpdir):
        # This is probably too basic to show that the wrapper works.
        filename = str(tmpdir.join('test.dat'))
        with io.open(filename, 'wb') as fw:
            fh = VLBIFileBase(fw)
            assert fh.fh_raw is fw
            assert not fh.readable()
            assert fh.writable()
            assert not fh.closed
            with pytest.raises(AttributeError):
                fh.bla
            assert repr(fh).startswith('VLBIFileBase(fh_raw')
            fh.write(b'abcd')
            fh.close()
            assert fh.closed
            assert fh.fh_raw.closed
        with io.open(filename, 'rb') as fr:
            fh = VLBIFileBase(fr)
            assert fh.fh_raw is fr
            assert fh.readable()
            assert not fh.writable()
            assert not fh.closed
            with pytest.raises(AttributeError):
                fh.bla
            assert repr(fh).startswith('VLBIFileBase(fh_raw')
            assert fh.read() == b'abcd'
            fh.close()
            assert fh.closed
            assert fh.fh_raw.closed


class TestSqueezeAndSubset(object):
    def setup(self):
        self.other_args = dict(fh_raw=None, header0=None, bps=1,
                               complex_data=False, samples_per_frame=1000,
                               sample_rate=10000*u.Hz, fill_value=0.,
                               verify=True)
        self.sample_shape_maker = namedtuple('SampleShape',
                                             'n0, n1, n2, n3, n4')
        self.unsliced_shape = (1, 21, 33, 1, 2)
        self.squeezed_shape = (21, 33, 2)
        self.squeezed_fields = ('n1', 'n2', 'n4')
        self.unsliced_data = np.ones((100,) + self.unsliced_shape, dtype='f4')
        self.squeezed_data = self.unsliced_data.squeeze()

    def make_reader_with_shape(self, squeeze=True, subset=None,
                               sample_shape_maker=None, unsliced_shape=None):

        class StreamReaderWithShape(VLBIStreamReaderBase):
            _sample_shape_maker = sample_shape_maker or self.sample_shape_maker

        return StreamReaderWithShape(
            unsliced_shape=unsliced_shape or self.unsliced_shape,
            squeeze=squeeze, subset=subset, **self.other_args)

    def make_writer_with_shape(self, squeeze=True, sample_shape_maker=None,
                               unsliced_shape=None):

        class StreamWriterWithShape(VLBIStreamWriterBase):
            _sample_shape_maker = sample_shape_maker or self.sample_shape_maker

        return StreamWriterWithShape(
            unsliced_shape=unsliced_shape or self.unsliced_shape,
            subset=None, squeeze=squeeze, **self.other_args)

    def test_sample_shape_and_squeeze(self):
        # Tests stream base's sample and squeezing routines.
        # Try tuple only.
        sb = VLBIStreamBase(unsliced_shape=self.unsliced_shape,
                            subset=None, squeeze=False, **self.other_args)
        assert sb.sample_shape == self.unsliced_shape
        sb = VLBIStreamBase(unsliced_shape=self.unsliced_shape,
                            subset=None, squeeze=True, **self.other_args)
        assert sb.sample_shape == self.squeezed_shape

        # Try reader with equivalent sample shape.
        sr = self.make_reader_with_shape(squeeze=False)
        assert sr.sample_shape == self.unsliced_shape
        assert sr.sample_shape._fields == self.sample_shape_maker._fields

        sr = self.make_reader_with_shape(squeeze=True)
        assert sr.sample_shape == self.squeezed_shape
        assert sr.sample_shape._fields == self.squeezed_fields
        assert (sr._squeeze_and_subset(self.unsliced_data).shape ==
                self.squeezed_data.shape)
        assert (sr._squeeze_and_subset(self.unsliced_data[:1]).shape ==
                (1,) + self.squeezed_shape)

        # With VLBIStreamWriterBase, we can access _unsqueeze.
        sw = self.make_writer_with_shape(squeeze=False)
        assert sw.sample_shape == self.unsliced_shape
        assert sw.sample_shape._fields == self.sample_shape_maker._fields

        sw = self.make_writer_with_shape(squeeze=True)
        assert sw.sample_shape == self.squeezed_shape
        assert sw.sample_shape._fields == self.squeezed_fields
        assert (sw._unsqueeze(self.squeezed_data).shape ==
                self.unsliced_data.shape)
        assert (sw._unsqueeze(self.squeezed_data[:1]).shape ==
                (1,) + self.unsliced_shape)

        # Check that single-axis sample shape squeezes to ().
        sample_shape_maker_s = namedtuple('SampleShape', 'n0')
        unsliced_shape_short = (1,)
        sws = self.make_writer_with_shape(False, sample_shape_maker_s,
                                          unsliced_shape_short)
        assert sws.sample_shape == (1,)
        sws = self.make_writer_with_shape(True, sample_shape_maker_s,
                                          unsliced_shape_short)
        assert sws.sample_shape == ()
        data = np.empty(100, dtype='float32')
        assert sws._unsqueeze(data).shape == (100, 1)

    @pytest.mark.parametrize(
        ('squeeze', 'subset', 'sliced_shape', 'sliced_n'),
        [(False, None, (1, 21, 33, 1, 2), ('n0', 'n1', 'n2', 'n3', 'n4')),
         (True, None, (21, 33, 2), ('n1', 'n2', 'n4')),
         (False, (), (1, 21, 33, 1, 2), ('n0', 'n1', 'n2', 'n3', 'n4')),
         (True, (), (21, 33, 2), ('n1', 'n2', 'n4')),
         (False, 0, (21, 33, 1, 2), ('n1', 'n2', 'n3', 'n4')),
         (True, 0, (33, 2), ('n2', 'n4')),
         (False, (0, 13), (33, 1, 2), ('n2', 'n3', 'n4')),
         (True, (0, 13), (2,), ('n4',)),
         (False, (Ellipsis, 0, 1), (1, 21, 33), None),
         (True, (Ellipsis, 0, 1), (21,), ('n1',)),
         (False, (0, slice(1, None, 4), slice(None), 0),
          (5, 33, 2), ('n1', 'n2', 'n4')),
         (True, (slice(1, None, 4), slice(None), [1]),
          (5, 33, 1), ('n1', 'n2', 'n4')),
         (False, (0, 0, slice(None, 1, -4)), (8, 1, 2), ('n2', 'n3', 'n4')),
         (True, (0, slice(None, 1, -4)), (8, 2), ('n2', 'n4')),
         (False, (0, np.array([2, 8, 9])[:, np.newaxis], [1, 7]),
          (3, 2, 1, 2), None),
         (True, (np.array([2, 8, 9])[:, np.newaxis], [1, 7], 0),
          (3, 2), None)])
    def test_squeeze_subset(self, squeeze, subset, sliced_shape, sliced_n):
        # Tests subsetting for squeezed samples.
        sb = self.make_reader_with_shape(squeeze=squeeze, subset=subset)
        if isinstance(subset, tuple):
            assert sb.subset == subset
        elif subset is None:
            assert sb.subset == ()
        else:
            assert sb.subset == (subset,)
        assert sb.sample_shape == sliced_shape
        assert getattr(sb.sample_shape, '_fields', None) == sliced_n
        subset_data = sb._squeeze_and_subset(self.unsliced_data)
        assert subset_data.shape == (100,) + sliced_shape

    def test_faulty_subset(self):
        # Advanced indexing changes dimensions, so sample_shape can't be set.
        with pytest.raises(IndexError) as excinfo:
            self.make_reader_with_shape(
                subset=([0], np.array([2, 8, 16])[:, np.newaxis], [1, 7]))
        assert "cannot be used to" in str(excinfo.value)

        # Can't subset with a string.
        with pytest.raises(IndexError) as excinfo:
            self.make_reader_with_shape(subset=(0, 'nonsense', [1, 7]))
        assert "cannot be used to" in str(excinfo.value)

        # Numerical index 8 is out of bounds of 3rd dimension
        with pytest.raises(IndexError) as excinfo:
            self.make_reader_with_shape(subset=(3, 0, [2, 8]))
        assert "cannot be used to" in str(excinfo.value)

        # Slice is out of bounds of 3rd dimension.
        with pytest.raises(AssertionError) as excinfo:
            self.make_reader_with_shape(subset=(3, 0, slice(4, 8)))


def test_crc():
    # Test example from age 4 of
    # http://www.haystack.mit.edu/tech/vlbi/mark5/docs/230.3.pdf
    stream = '0000 002D 0330 0000' + 'FFFF FFFF' + '4053 2143 3805 5'
    crc_expected = '284'
    crc12 = CRC(0x180f)
    stream = stream.replace(' ', '').lower()
    istream = int(stream, base=16)
    assert '{:037x}'.format(istream) == stream
    bitstream = np.array([((istream & (1 << bit)) != 0)
                          for bit in range(37*4-1, -1, -1)], np.bool)
    crcstream = crc12(bitstream)
    crc = np.bitwise_or.reduce(crcstream.astype(np.uint32) <<
                               np.arange(11, -1, -1))
    assert '{:03x}'.format(crc) == crc_expected
    fullstream = np.hstack((bitstream, crcstream))
    assert crc12.check(fullstream)


# PY2
@pytest.mark.parametrize(
    ('a', 'b', 'gcd_out'),
    ((7, 14, 7),
     (2712341, 234243, 1),
     (0, 5, 5),
     (4, -12, 4),
     (-4, -12, 4)))
def test_gcd(a, b, gcd_out):
    assert gcd(a, b) == gcd_out


@pytest.mark.parametrize(
    ('a', 'b', 'lcm_out'),
    ((7, 14, 14),
     (7853, 6199, 48680747),
     (0, 5, 0),
     (4, -12, 12),
     (-4, -12, 12)))
def test_lcm(a, b, lcm_out):
    assert lcm(a, b) == lcm_out
