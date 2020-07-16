# Licensed under the GPLv3 - see LICENSE
"""
Base definitions for baseband Headers, in particular for the VLBI types.

Defines a number of helpers to construct a Header class that hold header
words corresponding to a frame header, and provides access to the values
encoded in those words via a dict-like interface.  Definitions for headers
are constructed using a HeaderParser class (which is targeted specifically
at the VLBI formats).
"""
import sys
import struct
import warnings
import functools
from copy import copy

import numpy as np
from astropy.utils import sharedmethod

from .utils import fixedvalue


__all__ = ['four_word_struct', 'eight_word_struct',
           'make_parser', 'make_setter', 'get_default',
           'ParserDict', 'HeaderParserBase', 'HeaderParser',
           'ParsedHeaderBase', 'VLBIHeaderBase']


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
            return words[word_index] + (words[word_index + 1] << 32)

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
        bit_mask = (1 << bit_length) - 1
        if value is None:
            if default is None:
                raise ValueError("no default value so cannot set to 'None'.")
            value = default
        elif value is True:
            value = bit_mask
        elif np.any(value & bit_mask != value):
            raise ValueError("{0} cannot be represented with {1} bits"
                             .format(value, bit_length))
        if bit_length == 64:
            word1 = value & (1 << 32) - 1
            word2 = value >> 32
            words[word_index] = word1
            words[word_index + 1] = word2
        else:
            word = words[word_index]
            # Zero the part to be set, and add the value.
            bit_mask <<= bit_index
            word = ((word | bit_mask) ^ bit_mask) | (value << bit_index)
            words[word_index] = word
        return words

    return setter


def get_default(word_index, bit_index, bit_length, default=None):
    """Return the default value from a header keyword.

    Since it is called with the full description, it just returns
    the last item, defaulted to `None`.
    """
    return default


class ParserDict:
    """Create a lazily evaluated dictionary of parsers, setters, or defaults.

    Implemented as a non-data descriptor.  When first called on an instance,
    it will create a dict under the name of itself in the instance's
    ``__dict__``, which means that any further attribute access will return
    that dict instead of this descriptor.

    Parameters
    ----------
    function : callable
        Function that can be used to create a parser or setter, or get the
        default, based on a header keyword description.  Typically one of
        ``make_parser``, ``make_setter``, or ``get_default``.

    """

    def __init__(self, function):
        self.function = function

    def __set_name__(self, owner, name):
        self.name = name
        self.__doc__ = f"Lazily evaluated dict of {name}"

    def __get__(self, instance, cls=None):
        if instance is None:
            return self
        # Create dict of functions/defaults.
        d = {key: self.function(*definition)
             for key, definition in instance.items()}
        # Override ourselves on the instance.
        setattr(instance, self.name, d)
        return d

    def __repr__(self):
        return f"{self.__class__.__name__}({self.function}"


class HeaderParserBase(dict):
    """Parser & setter for header keywords.

    A dictionary of header keywords, with values that describe how they are
    encoded in a given header.  Initialisation is as a normal dict,
    with (ordered) key, value pairs, with each value a tuple containing
    information that describes how the value is encoded, and any default.

    The actual implementation is done by instances of
    `~baseband.base.header.ParserDict` called ``parsers``, ``setters``, and
    ``defaults``, which return functions that get a given keyword from
    header words, set the corresponding part of the header words to a value,
    or return the default value (if defined).  To speed up access to those,
    they are precalculated on first access rather than calculated on the fly.

    """
    def copy(self):
        return self.__class__(self)

    def __or__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        if sys.version_info >= (3, 9):  # pragma: no cover
            return self.__class__(super().__or__(other))

        result = self.__class__(self)
        result.update(other)
        return result

    # Backwards compatibility for code written for baseband < 4.0.
    __add__ = __or__

    def _clear_caches(self):
        """Clear the caches of the parser dicts. To be done on any change."""
        for key in set(self.__dict__):
            if isinstance(self.__class__.__dict__[key], ParserDict):
                del self.__dict__[key]


# Overwrite all dict methods that change the contents to clear the
# cache of the parsers and setters.
def make_wrapped_method(method):
    @functools.wraps(getattr(dict, method))
    def wrapped(self, *args, **kwargs):
        result = getattr(super(HeaderParserBase,
                               self), method)(*args, **kwargs)
        self._clear_caches()
        return result
    return wrapped


for method in ('__setitem__', 'update', 'pop', 'popitem', 'clear'):
    setattr(HeaderParserBase, method, make_wrapped_method(method))


class HeaderParser(HeaderParserBase):
    """Parser & setter for VLBI header keywords.

    A dictionary of header keywords, with values that describe how they are
    encoded in a given VLBI header.  Initialisation is as a normal dict,
    with (ordered) key, value pairs, with each value a tuple containing:

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
    return the default value (if defined).  To speed up access to those,
    they are precalculated on first access rather than calculated on the fly.

    """
    parsers = ParserDict(make_parser)
    setters = ParserDict(make_setter)
    defaults = ParserDict(get_default)


class ParsedHeaderBase:
    """Base class for all baseband headers using parsers.

    Defines a number of common routines.

    Generally, the actual class should define:

      _header_parser : HeaderParser instance corresponding to this class.

      _properties : tuple of properties accessible/usable in initialisation

    It also should define properties that tell the size (getters *and*
    setters, or use a `baseband.base.utils.fixedvalue` if the
    value is the same for all instances):

      payload_nbytes : number of bytes used by payload

      frame_nbytes : total number of bytes for header + payload

      get_time, set_time, and a corresponding time property :
           time at start of payload

    Parameters
    ----------
    words : tuple or list
        Header words.  If given as a tuple, the header is immutable.
    verify : bool, optional
        Whether to do basic verification of integrity.

    """

    _properties = ('payload_nbytes', 'frame_nbytes', 'time')
    """Properties accessible/usable in initialisation for all headers."""

    def __init__(self, words, verify=True):
        self.words = words
        if verify:
            self.verify()

    def verify(self):
        """Base verification always passes.

        Only here for subclasses to be able to do super().
        """
        pass  # pragma: no cover

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
        not_in_both = (set(self.keys()).symmetric_difference(kwargs)
                       - {'verify'})
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

    def update(self, *, verify=True, **kwargs):
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
        """Get the value of a particular header item from the header words."""
        try:
            return self._header_parser.parsers[item](self.words)
        except KeyError:
            raise KeyError("{0} header does not contain {1}"
                           .format(self.__class__.__name__, item))

    def __setitem__(self, item, value):
        """Set the value of a particular header item in the header words.

        If value is `None`, set the item to its default value (if it exists);
        if `True`, set all bits in the item (i.e., set item to its maximum).
        """
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
        """All keys defined for this header."""
        return self._header_parser.keys()

    def _ipython_key_completions_(self):
        # Enables tab-completion of header keys in IPython.
        return self.keys()

    def __contains__(self, key):
        return key in self.keys()

    def __eq__(self, other):
        return (type(self) is type(other)
                and np.all(np.array(self.words, copy=False)
                           == np.array(other.words, copy=False)))

    def _repr_value(self, key, value):
        return str(value)

    def __repr__(self):
        name = self.__class__.__name__
        outs = [f"{k}: {self._repr_value(k, self[k])}" for k in self.keys()]
        return "<{} {}>".format(name, (",\n  " + " "*len(name)).join(outs))


class VLBIHeaderBase(ParsedHeaderBase):
    """Base class for all VLBI headers.

    Defines a number of common routines.

    Generally, the actual class should define:

      _struct : `~struct.Struct` instance that can pack/unpack header words.

      _header_parser : `HeaderParser` instance corresponding to this class.

      _properties : tuple of properties accessible/usable in initialisation

      _invariants : set of keys of invariant header parts for a given type.

      _stream_invarants : set of keys of invariant header parts for a stream.

    It also should define properties that tell the size (getters *and*
    setters, or use a `baseband.base.utils.fixedvalue` if the
    value is the same for all instances):

      payload_nbytes : number of bytes used by payload

      frame_nbytes : total number of bytes for header + payload

      get_time, set_time, and a corresponding time property :
           time at start of payload

    Parameters
    ----------
    words : tuple or list of int, or None
        header words (generally, 32 bit unsigned int).  If given as a tuple,
        the header is immutable.  If `None`, set to a list of zeros for
        later initialisation (and skip any verification).
    verify : bool, optional
        Whether to do basic verification of integrity.  For the base class,
        checks that the number of words is consistent with the struct size.
    """

    # TODO: should [_stream]_invarants be defined through some subclass init??
    # TODO: perhaps from some hints in the headerparser definition?

    # Define a bare _struct to avoid sphinx complaints about nbytes.
    _struct = struct.Struct('')
    """Structure for the header words.  To be overridden by subclasses."""

    def __init__(self, words, verify=True):
        if words is None:
            words = [0] * (self._struct.size // 4)
            verify = False

        super().__init__(words, verify=verify)

    def verify(self):
        """Verify that the length of the words is consistent.

        Subclasses should override this to do more thorough checks.
        """
        assert len(self.words) == (self._struct.size // 4)

    @sharedmethod
    def invariants(self):
        """Set of keys of invariant header parts.

        On the class, this returns keys of parts that are shared by
        all headers for the type, on an instance, those that are
        shared with other headers in the same file.

        If neither are defined, returns 'sync_pattern' if the header
        containts that key.
        """

        if not isinstance(self, type) and hasattr(self, '_stream_invariants'):
            return self._stream_invariants

        elif hasattr(self, '_invariants'):
            return self._invariants

        elif 'sync_pattern' in getattr(self, '_header_parser', {}):
            return {'sync_pattern'}

        else:
            return set()

    @sharedmethod
    def invariant_pattern(self, invariants=None, **kwargs):
        """Pattern and mask shared between headers of a type or stream.

        This is mostly for use inside
        :meth:`~baseband.base.base.VLBIFileReaderBase.locate_frames`.

        Parameters
        ----------
        invariants : set of str, optional
            Set of keys to header parts that are shared between all headers
            of a given type or within a given stream/file.  Default: from
            `~baseband.base.header.VLBIHeaderBase.invariants()`.
        **kwargs
            Keyword arguments needed to instantiate an empty header.
            (Mostly for Mark 4).

        Returns
        -------
        pattern : list of int
            The pattern that is shared between headers. If called on
            an instance, just the header words; if called on a class,
            words with defaults for the relevant parts set.
        mask : list of int
            For each entry in ``pattern`` a bit mask with bits set for
            the parts that are invariant.
        """

        if invariants is None:
            invariants = self.invariants()

        if not invariants:
            raise ValueError("cannot create an invariant_mask without "
                             "some invariants")

        if isinstance(self, type):
            # If we are called as a classmethod, first get an instance
            # with all defaults set.  This will be our pattern.
            self = self(None, **kwargs)
            for invariant in invariants:
                value = self._header_parser.defaults[invariant]
                if value is None:
                    raise ValueError('can only set as invariant a header '
                                     'part that has a default.')
                self[invariant] = value

        # Create an all-zero version and set bits for all invariants.
        mask = self.__class__(None, **kwargs)
        for invariant in invariants:
            mask[invariant] = True

        return self.words, mask.words

    @fixedvalue
    def nbytes(cls):
        """Size of the header in bytes."""
        return cls._struct.size

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

    def _repr_value(self, key, value):
        if key.startswith(('bcd', 'crc', 'sync_pattern')):
            try:
                value = hex(value)
            except Exception:
                pass
        return super()._repr_value(key, value)
