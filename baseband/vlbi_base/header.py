# Helper functions for VLBI readers (VDIF, Mark5B).
import struct
import warnings

from astropy.utils import OrderedDict

OPTIMAL_2BIT_HIGH = 3.3359
eight_word_struct = struct.Struct('<8I')
four_word_struct = struct.Struct('<4I')


def make_parser(word_index, bit_index, bit_length):
    """Convert specific bits from a header word to a bool or integer."""
    if bit_length == 1:
        return lambda words: bool((words[word_index] >> bit_index) & 1)
    elif bit_length == 32:
        assert bit_index == 0
        return lambda words: words[word_index]
    else:
        mask = (1 << bit_length) - 1  # e.g., bit_length=8 -> 0xff
        if bit_index == 0:
            return lambda words: words[word_index] & mask
        else:
            return lambda words: (words[word_index] >> bit_index) & mask


def make_setter(word_index, bit_index, bit_length, default=None):
    def setter(words, value):
        if value is None and default is not None:
            value = default
        value = int(value)
        word = words[word_index]
        bit_mask = (1 << bit_length) - 1
        # Check that value will fit within the bit limits.
        if value & bit_mask != value:
            raise ValueError("{0} cannot be represented with {1} bits"
                             .format(value, bit_length))
        # Zero the part to be set.
        bit_mask <<= bit_index
        word = (word | bit_mask) ^ bit_mask
        # Add the value
        word |= value << bit_index
        return words[:word_index] + (word,) + words[word_index+1:]
    return setter


class HeaderPropertyGetter(object):
    def __init__(self, getter):
        self.getter = getter

    def __get__(self, instance, owner_cls):
        return HeaderProperty(instance, self.getter)


class HeaderProperty(object):
    """Mimic a dictionary, calculating entries from header words."""
    def __init__(self, header_parser, getter):
        self.header_parser = header_parser
        self.getter = getter

    def __getitem__(self, item):
        definition = self.header_parser[item]
        return self.getter(definition)

    def __getattr__(self, attr):
        try:
            return super(HeaderProperty, self).__getattribute__(attr)
        except AttributeError:
            return getattr(self.header_parser)


class HeaderParser(OrderedDict):
    def __init__(self, *args, **kwargs):
        super(HeaderParser, self).__init__(*args, **kwargs)
        # In principle, we could calculate the parsers on the fly,
        # like we do for the setters, but this would be needlessly slow,
        # so we precalculate all of them, using a dict for even better speed.
        self.parsers = {k: make_parser(*v[:3]) for k, v in self.items()}

    def __add__(self, other):
        if not isinstance(other, HeaderParser):
            return NotImplemented
        result = self.copy()
        result.update(other)
        return result

    defaults = HeaderPropertyGetter(
        lambda definition: definition[3] if len(definition) > 3 else None)

    setters = HeaderPropertyGetter(
        lambda definition: make_setter(*definition))

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

      _struct: HeaderParser instance corresponding to this class.
      _header_parser: HeaderParser instance corresponding to this class.

    It also should define properties (getters *and* setters):

      payloadsize: number of bytes used by payload

      framesize: total number of bytes for header + payload

      get_time, set_time, and a corresponding time property:
           time at start of payload
    """

    def __init__(self, words, verify=True):
        if words is None:
            self.words = (0,) * (self._struct.size // 4)
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
        return self.__class__(self.words, verify=False)

    @property
    def size(self):
        return self._struct.size

    @classmethod
    def fromfile(cls, fh, *args, **kwargs):
        """Read VLBI Header from file.

        Arguments are the same as for class initialisation.
        """
        size = cls._struct.size
        s = fh.read(size)
        if len(s) != size:
            raise EOFError
        return cls(cls._struct.unpack(s), *args, **kwargs)

    def tofile(self, fh):
        """Write VLBI Frame header to filehandle."""
        return fh.write(self._struct.pack(*self.words))

    @classmethod
    def fromvalues(cls, *args, **kwargs):
        """Initialise a header from parsed values.

        Here, the parsed values must be given as keyword arguments, i.e.,
        for any header = cls(<somedata>), cls.fromvalues(**header) == header.

        However, unlike for the 'fromkeys' class method, data can also be set
        using arguments named after header methods such 'time'.

        If any arguments are needed to initialize an empty header, those
        can be passed on in ``*args``.
        """
        # Initialize an empty header.
        self = cls(None, *args, verify=False)
        # First set all keys to keyword arguments or defaults.
        for key in self.keys():
            if key in kwargs:
                self[key] = kwargs.pop(key)
            elif self._header_parser.defaults[key] is not None:
                self[key] = self._header_parser.defaults[key]

        # Next, use remaining keyword arguments to set properties.
        # Order may be important so use list:
        for key in self._properties:
            if key in kwargs:
                setattr(self, key, kwargs.pop(key))

        if kwargs:
            warnings.warn("Some keywords unused in header initialisation: {0}"
                          .format(kwargs))
        self.verify()
        return self

    @classmethod
    def fromkeys(cls, *args, **kwargs):
        """Like fromvalues, but without any interpretation of keywords."""
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
        try:
            return self._header_parser.parsers[item](self.words)
        except KeyError:
            raise KeyError("{0} header does not contain {1}"
                           .format(self.__class__.__name__, item))

    def __setitem__(self, item, value):
        try:
            self.words = self._header_parser.setters[item](self.words, value)
        except KeyError:
            raise KeyError("{0} header does not contain {1}"
                           .format(self.__class__.__name__, item))

    def __getattr__(self, attr):
        try:
            return super(VLBIHeaderBase, self).__getattribute__(attr)
        except AttributeError:
            if attr in self.keys():
                return self[attr]
            else:
                raise

    def keys(self):
        return self._header_parser.keys()

    def __eq__(self, other):
        return (type(self) is type(other) and
                list(self.words) == list(other.words))

    def __contains__(self, key):
        return key in self.keys()

    def __repr__(self):
        name = self.__class__.__name__
        return ("<{0} {1}>".format(name, (",\n  " + len(name) * " ").join(
            ["{0}: {1}".format(k, self[k]) for k in self.keys()])))
