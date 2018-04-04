# Licensed under the GPLv3 - see LICENSE
"""
Base definitions for VLBI Headers, used for VDIF and Mark 5B.

Defines a header class VLBIHeaderBase that can be used to hold the words
corresponding to a frame header, providing access to the values encoded in
via a dict-like interface.  Definitions for headers are constructed using
the HeaderParser class.
"""
from __future__ import absolute_import, division, print_function
from copy import copy
import struct
import warnings
from collections import OrderedDict
import numpy as np


__all__ = ['four_word_struct', 'eight_word_struct',
           'make_parser', 'make_setter',
           'HeaderProperty', 'HeaderPropertyGetter',
           'HeaderParser', 'VLBIHeaderBase']

four_word_struct = struct.Struct('<4I')
"""Struct instance that packs/unpacks 4 unsigned 32-bit integers."""
eight_word_struct = struct.Struct('<8I')
"""Struct instance that packs/unpacks 8 unsigned 32-bit integers."""


def make_parser(word_index, bit_index, bit_length, default=None):
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
            return words[word_index] + words[word_index + 1] * (1 << 32)

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
            words[word_index] = word1
            words[word_index + 1] = word2
            return words

        word = words[word_index]
        # Zero the part to be set, and add the value.
        bit_mask <<= bit_index
        word = ((word | bit_mask) ^ bit_mask) | (value << bit_index)
        words[word_index] = word
        return words

    return setter


def get_default(word_index, bit_index, bit_length, default=None):
    return default


class HeaderProperty(object):
    """Mimic a dictionary, calculating entries from header words.

    Used to calculate setter functions and extract default values.

    Parameters
    ----------
    header_parser : `HeaderParser`
        A dict with header encoding information.
    getter : function
        Function that uses the encoding information to calculate a result.
    """
    def __init__(self, header_parser, getter, doc=None):
        self.header_parser = header_parser
        self.getter = getter
        if doc is not None:
            self.__doc__ = doc

    def __getitem__(self, item):
        definition = self.header_parser[item]
        return self.getter(*definition)


class HeaderPropertyGetter(object):
    """Special property for attaching HeaderProperty."""
    def __init__(self, getter, doc=None):
        self.getter = getter
        self.__doc__ = doc or getter.__doc__

    def __get__(self, instance, owner_cls=None):
        if instance is None:  # pragma: no cover
            return self
        return HeaderProperty(instance, getattr(instance, self.getter),
                              doc=self.__doc__)


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
        self._make_parser = kwargs.pop('make_parser', make_parser)
        self._make_setter = kwargs.pop('make_setter', make_setter)
        self._get_default = kwargs.pop('get_default', get_default)
        # Use a dict rather than OrderedDict for the parsers for better speed.
        # Note that this gets filled by calls to __setitem__.
        self._parsers = {}
        super(HeaderParser, self).__init__(*args, **kwargs)

    def copy(self):
        """Make an independent copy."""
        return self.__class__(self, make_parser=self._make_parser,
                              make_setter=self._make_setter,
                              get_default=self._get_default)

    def __add__(self, other):
        if not isinstance(other, HeaderParser):
            return NotImplemented
        result = self.copy()
        result.update(other)
        return result

    def __setitem__(self, item, value):
        self._parsers[item] = self._make_parser(*value)
        super(HeaderParser, self).__setitem__(item, value)

    @property
    def parsers(self):
        """Dict with functions to get specific header values."""
        return self._parsers

    defaults = HeaderPropertyGetter(
        '_get_default',
        doc="Dict-like allowing access to default header values.")

    setters = HeaderPropertyGetter(
        '_make_setter',
        doc="Dict-like returning function to set specific header value.")

    def update(self, other):
        """Update the parser with the information from another one."""
        if not isinstance(other, HeaderParser):
            raise TypeError("can only update using a HeaderParser instance.")
        super(HeaderParser, self).update(other)
        # Update the parsers rather than recalculate all the functions.
        self._parsers.update(other._parsers)


class VLBIHeaderBase(object):
    """Base class for all VLBI headers.

    Defines a number of common routines.

    Generally, the actual class should define:

      _struct: `~struct.Struct` instance that can pack/unpack header words.

      _header_parser: `HeaderParser` instance corresponding to this class.

      _properties: tuple of properties accessible/usable in initialisation

    It also should define properties (getters *and* setters):

      payload_nbytes: number of bytes used by payload

      frame_nbytes: total number of bytes for header + payload

      get_time, set_time, and a corresponding time property:
           time at start of payload

    Parameters
    ----------
    words : tuple or list of int, or None
        header words (generally, 32 bit unsigned int).  If `None`,
        set to a list of zeros for later initialisation.  If given as a tuple,
        the header is immutable.
    verify : bool
        Whether to do basic verification of integrity.  For the base class,
        checks that the number of words is consistent with the struct size.
    """

    _properties = ('payload_nbytes', 'frame_nbytes', 'time')
    """Properties accessible/usable in initialisation for all headers."""

    def __init__(self, words, verify=True):
        if words is None:
            self.words = [0] * (self._struct.size // 4)
        else:
            self.words = words
        if verify:
            self.verify()

    def verify(self):
        """Verify that the length of the words is consistent.

        Subclasses should override this to do more thorough checks.
        """
        assert len(self.words) == (self._struct.size // 4)

    def copy(self, **kwargs):
        """Create a mutable and independent copy of the header.

        Keyword arguments can be passed on as needed by possible subclasses.
        """
        kwargs.setdefault('verify', False)
        new = self.__class__(copy(self.words), **kwargs)
        new.mutable = True
        return new

    def __copy__(self):
        return self.copy()

    @property
    def nbytes(self):
        """Size of the header in bytes."""
        return self._struct.size

    @property
    def mutable(self):
        """Whether the header can be modified."""
        if isinstance(self.words, tuple):
            return False
        word0 = self.words[0]
        try:
            self.words[0] = 0
        except Exception:
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
            raise TypeError("do not know how to set mutability of '.words' "
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

        Here, the parsed values must be given as keyword arguments, i.e., for
        any ``header = cls(<words>)``, ``cls.fromvalues(**header) == header``.

        However, unlike for the `fromkeys` class method, data can also be set
        using arguments named after header methods, such as ``time``.

        Parameters
        ----------
        *args
            Possible arguments required to initialize an empty header.
        **kwargs
            Values used to initialize header keys or methods.
        """
        # Initialize an empty header, and update it with the keyword arguments.
        self = cls(None, *args, verify=False)
        # Set defaults in keyword arguments.
        for key in set(self.keys()).difference(kwargs.keys()):
            default = self._header_parser.defaults[key]
            if default is not None:
                kwargs[key] = default

        self.update(**kwargs)
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
        not_in_both = (set(self.keys()).symmetric_difference(kwargs) -
                       {'verify'})
        if not_in_both:
            not_in_kwarg = set(self.keys()).difference(kwargs)
            not_in_self = set(kwargs).difference(self.keys()) - {'verify'}
            msg_parts = []
            for item, msg in ((not_in_kwarg, "is missing keywords ({0})"),
                              (not_in_self, "contains extra keywords ({0})")):
                if item:
                    msg_parts.append(msg.format(item))

            raise KeyError("input list " + " and ".join(msg_parts))

        self.update(**kwargs)
        return self

    def update(self, **kwargs):
        """Update the header by setting keywords or properties.

        Here, any keywords matching header keys are applied first, and any
        remaining ones are used to set header properties, in the order set
        by the class (in ``_properties``).

        Parameters
        ----------
        verify : bool, optional
            If `True` (default), verify integrity after updating.
        **kwargs
            Arguments used to set keywords and properties.
        """
        verify = kwargs.pop('verify', True)

        # First use keywords which are also keys into self.
        for key in set(kwargs.keys()).intersection(self.keys()):
            self[key] = kwargs.pop(key)

        # Next, use remaining keyword arguments to set properties.
        # Order is important, so we cannot use an intersection as above.
        if kwargs:
            for key in self._properties:
                if key in kwargs:
                    setattr(self, key, kwargs.pop(key))

            if kwargs:
                warnings.warn("some keywords unused in header update: {0}"
                              .format(kwargs))

        if verify:
            self.verify()

    def __getitem__(self, item):
        """Get the value a particular header item from the header words."""
        try:
            return self._header_parser._parsers[item](self.words)
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
                raise TypeError("header is immutable. Set '.mutable` attribute"
                                " or make a copy.")
            else:
                raise

    def keys(self):
        return self._header_parser.keys()

    def _ipython_key_completions_(self):
        # Enables tab-completion of header keys in IPython.
        return self.keys()

    def __contains__(self, key):
        return key in self.keys()

    def __eq__(self, other):
        return (type(self) is type(other) and
                np.all(np.array(self.words, copy=False) ==
                       np.array(other.words, copy=False)))

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
