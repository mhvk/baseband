"""
Base definitions for VLBI Headers, used for VDIF and Mark 5B.

Defines a header class VLBIHeaderBase that can be used to hold the words
corresponding to a frame header, providing access to the values encoded in
via a dict-like interface.  Definitions for headers are constructed using
the HeaderParser class.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import struct
import warnings
import numpy as np

from astropy.utils import OrderedDict

OPTIMAL_2BIT_HIGH = 3.3359
eight_word_struct = struct.Struct('<8I')
four_word_struct = struct.Struct('<4I')


def make_parser(word_index, bit_index, bit_length):
    """Construct a function that converts specific bits from a header.

    The function acts on a tuple/array of 32-bit words, extracting given bits
    from a specific word and convert them to bool (for single bit) or integer.

    The parameters are those that define header keywords, and all parsers do
    ``(words[word_index] >> bit_index) & ((1 << bit_length) - 1)``, except that
    that they have been optimized for the specific cases of single bits,
    full words, and items starting at bit 0.  As a special case, bit_length=64
    allows one to extract two words as a single (long) integer.

    Parameters
    ----------
    word_index : int
        Index into the tuple of words passed to the function.
    bit_index : int
        Index to the starting bit of the part to be extracted.
    bit_length : int
        Number of bits to be extracted.

    Returns
    -------
    parser : function
        To be used as ``parser(words)``.
    """
    if bit_length == 1:
        def parser(words):
            return (words[word_index] & (1 << bit_index)) != 0

    elif bit_length == 32:
        assert bit_index == 0

        def parser(words):
            return words[word_index]

    elif bit_length == 64:
        assert bit_index == 0

        def parser(words):
            return words[word_index] + words[word_index+1] * (1 << 32)

    else:
        bit_mask = (1 << bit_length) - 1  # e.g., bit_length=8 -> 0xff
        if bit_index == 0:
            def parser(words):
                return words[word_index] & bit_mask

        else:
            def parser(words):
                return (words[word_index] >> bit_index) & bit_mask

    return parser


def make_setter(word_index, bit_index, bit_length, default=None):
    """Construct a function that uses a value to set specific bits in a header.

    The function will act on a tuple/array of words, setting given bits
    from a given word using a value.

    The parameters are just those that define header keywords.

    Parameters
    ----------
    word_index : int
        Index into the tuple of words passed to the function.
    bit_index : int
        Index to the starting bit of the part to be extracted.
    bit_length : int
        Number of bits to be extracted.
    default : int or bool or None
        Possible default value to use in function if no default is passed on.

    Returns
    -------
    setter : function
        To be used as ``setter(words, value)``.
    """
    def setter(words, value):
        if value is None and default is not None:
            value = default
        bit_mask = (1 << bit_length) - 1
        # Check that value will fit within the bit limits.
        if np.any(value & bit_mask != value):
            raise ValueError("{0} cannot be represented with {1} bits"
                             .format(value, bit_length))
        if bit_length == 64:
            word1 = value & (1 << 32) - 1
            word2 = value >> 32
            words[word_index:word_index+1] = word1
            words[word_index+1:word_index+2] = word2
            return words

        word = words[word_index]
        # Zero the part to be set.
        bit_mask <<= bit_index
        word = ((word | bit_mask) ^ bit_mask)
        # Add the value
        word |= value << bit_index
        words[word_index] = word
        return words

    return setter


class HeaderProperty(object):
    """Mimic a dictionary, calculating entries from header words.

    Used below to calculate setter functions and extract default values.

    Parameters
    ----------
    header_parser : HeaderParser instance
        An dict with header encoding information.
    getter : function
        Function that uses the encoding information to calculate a result.
    """
    def __init__(self, header_parser, getter):
        self.header_parser = header_parser
        self.getter = getter

    def __getitem__(self, item):
        definition = self.header_parser[item]
        return self.getter(definition)


class HeaderPropertyGetter(object):
    """Special property for attaching HeaderProperty."""
    def __init__(self, getter, doc=None):
        self.getter = getter
        if doc is None and getter.__doc__ is not None:
            doc = getter.__doc__
        self.__doc__ = doc

    def __get__(self, instance, owner_cls=None):
        return HeaderProperty(instance, self.getter)


class HeaderParser(OrderedDict):
    """Parser & setter for VLBI header keywords.

    An ordered dict of header keywords, with values that describe how they are
    encoded in a given VLBI header.  Initialisation is as a normal OrderedDict,
    with a key, value pairs.  The value should be a tuple containing:

    word_index : int
        Index into the header words for this key.
    bit_index : int
        Index to the starting bit of the part used for this key.
    bit_length : int
        Number of bits.
    default : int or bool or None
        Possible default value to use in initialisation (e.g., a sync pattern).

    The class provides dict-like properties ``parsers``, ``setters``, and
    ``defaults``, which return functions that get a given keyword from header
    words, set the corresponding part of the header words to a value, or
    return the default value (if defined).

    Note that while in principle, parsers and setters could be calculated on
    the fly, we precalculate the parsers to speed up header keyword access.
    """
    def __init__(self, *args, **kwargs):
        # Use a dict rather than OrderedDict for the parsers for better speed.
        # Note that this gets filled by calls to __setitem__.
        self.parsers = {}
        super(HeaderParser, self).__init__(*args, **kwargs)
        self.parsers = {k: make_parser(*v[:3]) for k, v in self.items()}

    def __add__(self, other):
        if not isinstance(other, HeaderParser):
            return NotImplemented
        result = self.copy()
        result.update(other)
        return result

    def __setitem__(self, item, value):
        self.parsers[item] = make_parser(*value[:3])
        super(HeaderParser, self).__setitem__(item, value)

    defaults = HeaderPropertyGetter(
        lambda definition: definition[3] if len(definition) > 3 else None,
        doc="Dict-like allowing access to default header values by keyword.")

    setters = HeaderPropertyGetter(
        lambda definition: make_setter(*definition),
        doc="Dict-like returning function to set header keyword to a value")

    def update(self, other):
        if not isinstance(other, HeaderParser):
            raise TypeError("Can only update using a HeaderParser instance.")
        super(HeaderParser, self).update(other)
        # Update the parsers rather than recalculate all the functions.
        self.parsers.update(other.parsers)


class VLBIHeaderBase(object):
    """Base class for all VLBI headers.

    Defines a number of common routines.

    Generally, the actual class should define:

      _struct: `~struct.Struct` instance that can pack/unpack header words.

      _header_parser: HeaderParser instance corresponding to this class.

    It also should define properties (getters *and* setters):

      payloadsize: number of bytes used by payload

      framesize: total number of bytes for header + payload

      get_time, set_time, and a corresponding time property:
           time at start of payload

    Parameters
    ----------
    words : tuple or list of int, or None
        header words (generally, 32 bit unsigned int).  If ``None``,
        set to a list of zeros for later initialisation.  If given as a tuple,
        the header is immutable.
    verify : bool
        Whether to do basic verification of integrity.  For the base class,
        checks that the number of words is consistent with the struct size.
    """
    def __init__(self, words, verify=True):
        if words is None:
            self.words = [0,] * (self._struct.size // 4)
        else:
            self.words = words
        if verify:
            self.verify()

    def verify(self):
        """Verify that the length of the words is consistent.

        Subclasses should override this to do more thorough checks.
        """
        assert len(self.words) == (self._struct.size // 4)

    def copy(self):
        """Copy the header.

        Any tuple of words is converted to a list so that they can be changed.
        """
        if isinstance(self.words, tuple):
            words = list(self.words)
        else:
            words = self.words.copy()
        return self.__class__(words, verify=False)

    def __copy__(self):
        return self.copy()

    @property
    def size(self):
        """Size of the header in bytes."""
        return self._struct.size

    @property
    def mutable(self):
        if isinstance(self.words, tuple):
            return False
        word0 = self.words[0]
        try:
            self.words[0] = 0
        except:
            return False
        else:
            self.words[0] = word0
            return True

    @mutable.setter
    def mutable(self, mutable):
        if isinstance(self.words, np.ndarray):
            self.words.flags['WRITEABLE'] = mutable
        elif isinstance(self.words, tuple):
            if mutable:
                self.words = list(self.words)
        elif isinstance(self.words, list):
            if not mutable:
                self.words = tuple(self.words)
        else:
            raise TypeError("Do not know how to set mutability of '.words' "
                            "of class {0}".format(type(self.words)))

    @classmethod
    def fromfile(cls, fh, *args, **kwargs):
        """Read VLBI Header from file.

        Arguments are the same as for class initialisation.  The header
        constructed will be immutable.
        """
        s = fh.read(cls._struct.size)
        if len(s) != cls._struct.size:
            raise EOFError
        return cls(cls._struct.unpack(s), *args, **kwargs)

    def tofile(self, fh):
        """Write VLBI frame header to filehandle."""
        return fh.write(self._struct.pack(*self.words))

    @classmethod
    def fromvalues(cls, *args, **kwargs):
        """Initialise a header from parsed values.

        Here, the parsed values must be given as keyword arguments, i.e.,
        for any header = cls(<somewords>), cls.fromvalues(**header) == header.

        However, unlike for the 'fromkeys' class method, data can also be set
        using arguments named after header methods such 'time'.

        If any arguments are needed to initialize an empty header, those
        can be passed on in ``*args``.
        """
        verify = kwargs.pop('verify', True)
        # Initialize an empty header.
        self = cls(None, *args, verify=False)
        # First set all keys to keyword arguments or defaults.
        for key in self.keys():
            value = kwargs.pop(key, self._header_parser.defaults[key])
            if value is not None:
                self[key] = value

        # Next, use remaining keyword arguments to set properties.
        # Order may be important so use list:
        for key in self._properties:
            if key in kwargs:
                setattr(self, key, kwargs.pop(key))

        if kwargs:
            warnings.warn("Some keywords unused in header initialisation: {0}"
                          .format(kwargs))
        if verify:
            self.verify()
        return self

    @classmethod
    def fromkeys(cls, *args, **kwargs):
        """Initialise a header from parsed values.

        Like fromvalues, but without any interpretation of keywords.

        Raises
        ------
        KeyError : if not all keys required are present in ``kwargs``
        """
        self = cls(None, *args, verify=False)
        for key in self.keys():
            self.words = self._header_parser.setters[key](
                self.words, kwargs.pop(key))

        if kwargs:
            warnings.warn("Some keywords unused in header initialisation: {0}"
                          .format(kwargs))
        self.verify()
        return self

    def __getitem__(self, item):
        """Get the value a particular header item from the header words."""
        try:
            return self._header_parser.parsers[item](self.words)
        except KeyError:
            raise KeyError("{0} header does not contain {1}"
                           .format(self.__class__.__name__, item))

    def __setitem__(self, item, value):
        """Set the value of a particular header item in the header words."""
        try:
            self._header_parser.setters[item](self.words, value)
        except KeyError:
            raise KeyError("{0} header does not contain {1}"
                           .format(self.__class__.__name__, item))
        except(TypeError, ValueError):
            if not self.mutable:
                raise TypeError("Header is immutable. Set '.mutable` attribute"
                                " or make a copy.")
            else:
                raise

    def __getattr__(self, attr):
        """Get attribute, or, failing that, try to get key from header."""
        try:
            return super(VLBIHeaderBase, self).__getattribute__(attr)
        except AttributeError:
            if attr in self.keys():
                return self[attr]
            else:
                raise

    def keys(self):
        return self._header_parser.keys()

    def __contains__(self, key):
        return key in self.keys()

    def __eq__(self, other):
        return (type(self) is type(other) and
                list(self.words) == list(other.words))

    @staticmethod
    def _repr_as_hex(key):
        return (key.startswith('bcd') or key.startswith('crc') or
                key == 'sync_pattern')

    def __repr__(self):
        name = self.__class__.__name__
        return ("<{0} {1}>".format(name, (",\n  " + len(name) * " ").join(
            ["{0}: {1}".format(k, hex(self[k]) if self._repr_as_hex(k)
                               else self[k])
             for k in self.keys()])))
