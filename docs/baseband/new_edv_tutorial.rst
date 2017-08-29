.. _new_edv:

*************************
Supporting a New VDIF EDV
*************************

Users may encounter VDIF files with unusual headers not currently supported
by Baseband.  These may either have novel EDV, or they may purport to be a
supported EDV but not conform to its specification on the 
`VDIF website <http://www.vlbi.org/vdif/>`_.  Baseband supports easy
implementation of new EDVs and overriding of exisiting EDVs, without
the need to modify Baseband's source code, to handle such situations.

The tutorials below assumes the following modules have been imported::

    >>> import numpy as np
    >>> from baseband import vdif, vlbi_base as vlbi

.. _new_edv_vdif_headers:

VDIF Headers
============

Each VDIF frame begins with a 32-byte, or eight 32-bit **word**,
header that is structured as follows:

.. figure:: VDIFHeader.png
   :scale: 50 %

   Schematic of the standard 32-bit VDIF header, from `VDIF specification 
   release 1.1.1 document, Fig. 3
   <http://www.vlbi.org/vdif/docs/VDIF_specification_Release_1.1.1.pdf>`_.
   32-bit words are labelled on the left, while byte and bit numbers above
   indicate relative addresses within each word.  Subscripts indicate field
   length in bits.

where the abbreviated labels are

- :math:`\mathrm{I}_1` - invalid data
- :math:`\mathrm{L}_1` - if 1, header is VDIF legacy
- :math:`\mathrm{V}_3` - VDIF version number
- :math:`\mathrm{log}_2\mathrm{(\#chns)}_5` - :math:`\mathrm{log}_2` of the
  number of sub-bands in the frame
- :math:`\mathrm{C}_1` - if 1, complex data
- :math:`\mathrm{EDV}_8` - "extended data version" number; see below

Detailed definitions of terms are found on pg. 5 - 7 of the `VDIF specification
document <http://www.vlbi.org/vdif/docs/VDIF_specification_Release_1.1.1.pdf>`_.

Words 4 - 7 hold optional extended user data that is telescope or experiment-
specific.  The layout of this data is specified by the EDV, in word 4
of the header.  EDV formats can be registered on the `VDIF website
<http://www.vlbi.org/vdif/>`_; all registered formats there are supported by
Baseband except for EDV = 4.

.. _new_edv_new_edv:

Implementing a New EDV
======================

In this tutorial, we follow the implementation of an EDV = 4 header,
designed to independently store the validity of sub-band channels within a
single data frame, rather than using the single bit indicating valid/invalid
data in the default VDIF header.  Its header is given by 
`its specification <http://www.vlbi.org/vdif/docs/edv4description.pdf>`_.
Specifically, we need to add to the standard VDIF header:

- Validity header mask (word 4, bits 16 - 24): integer value between 1 and
  64 inclusive indicating the number of validity bits.  (This is different
  than :math:`\mathrm{log}_2\mathrm{(\#chns)}_5`, since some channels can be
  unused.)
- Synchronization pattern (word 5): constant byte sequence ``0xACABFEED``,
  for finding the locations of headers in a data stream.
- Validity mask (words 6 - 7): 64-bit binary mask indicating the validity of
  sub-bands.  Any fraction of 64 sub-bands can be stored in this format,
  with any unused bands labelled as invalid (``0``) in the mask.  If the
  number of bands exceeds 64, each bit indicates the validity of a group
  of sub-bands; see specification for details.

See Sec. 3.1 of the specification for best practices on using
the invalid data bit :math:`\mathrm{I}_1` in word 0.

In Baseband, a header is parsed using :class:`~baseband.vdif.header.VDIFHeader`,
which returns a header object.  The object's type is a subclass of
:class:`~!baseband.vdif.header.VDIFHeader` that corresponds to the header
EDV.  This can be seen in the :mod:`~baseband.vdif.header` module class
inheritance diagram.  To support a new EDV, we create a new subclass to
:class:`~!baseband.vdif.header.VDIFHeader`::

    >>> class VDIFHeader4(vdif.header.VDIFHeader):
    ...     _edv = 4
    ...     
    ...     _header_parser = vlbi.header.HeaderParser(
    ...         (('invalid_data', (0, 31, 1, False)),
    ...          ('legacy_mode', (0, 30, 1, False)),
    ...          ('seconds', (0, 0, 30)),
    ...          ('_1_30_2', (1, 30, 2, 0x0)),
    ...          ('ref_epoch', (1, 24, 6)),
    ...          ('frame_nr', (1, 0, 24, 0x0)),
    ...          ('vdif_version', (2, 29, 3, 0x1)),
    ...          ('lg2_nchan', (2, 24, 5)),
    ...          ('frame_length', (2, 0, 24)),
    ...          ('complex_data', (3, 31, 1)),
    ...          ('bits_per_sample', (3, 26, 5)),
    ...          ('thread_id', (3, 16, 10, 0x0)),
    ...          ('station_id', (3, 0, 16)),
    ...          ('edv', (4, 24, 8)),
    ...          ('valid_mask_length', (4, 16, 8, 0)),
    ...          ('sync_pattern', (5, 0, 32, 0xACABFEED)),
    ...          ('valid_mask_high', (6, 0, 32, 0)),
    ...          ('valid_mask_low', (7, 0, 32, 0))))

:class:`~!baseband.vdif.header.VDIFHeader` is metaclassed such that whenever
it is subclassed, the subclass definition is inserted into the
:obj:`~baseband.vdif.header.VDIF_HEADER_CLASSES` dictionary and indexed
by its EDV value.  Methods in :class:`~!baseband.vdif.header.VDIFHeader` use
this dictionary to determine the type of object to return for a particular
EDV.  How all this works is further discussed in the VDIF
:mod:`header <baseband.vdif.header>` module documentation.

The class must have a private ``_edv`` attribute for it to properly be
registered in :obj:`~!baseband.vdif.header.VDIF_HEADER_CLASSES`.  It must
also feature a ``_header_parser`` that reads these words to return header
properties.  For this, we utilize :class:`vlbi_base.header.HeaderParser
<baseband.vlbi_base.header.HeaderParser>`, available in 
:mod:`baseband.vlbi_base.header`.  To initialize a header parser,
we pass it a tuple of header property keys, where each key follows the
syntax:

    ``('property_name', (word_index, bit_index, bit_length, default))``

where

- ``property_name``: name of the header property key
- ``word_index``: index into the header words for the key
- ``bit_index``: index to the starting bit of the part used
  for the key
- ``bit_length``: number of bits used by the key
- ``default``: (optional) default value to use in initialisation

For further details, see the :class:`~baseband.vlbi_base.header.HeaderParser`
documentation.

Once defined, we can use our new header like any other::

    >>> myheader = vdif.header.VDIFHeader.fromvalues(
    ...     edv=4, seconds=14363767, nchan=1,
    ...     station=65532, bps=2, complex_data=False,
    ...     thread_id=3, valid_mask_length=64,
    ...     valid_mask_high=(1 << 31),
    ...     valid_mask_low=(1 << 28))
    >>> isinstance(myheader, VDIFHeader4)
    True
    >>> myheader['station_id'] == 65532
    True
    >>> myheader['sync_pattern'] == 0xACABFEED
    True
    >>> myheader['valid_mask_high'] == 2**31
    True
    >>> myheader['valid_mask_low'] == 2**28
    True

There is an easier means of instantiating the header parser.  As can be
seen in :mod:`~baseband.vdif.header` class inheritance diagram, many VDIF
headers are subclassed from other :mod:`~baseband.vdif.header.VDIFHeader`
subclasses, namely :mod:`~baseband.vdif.header.VDIFBaseHeader` and
:mod:`~baseband.vdif.header.VDIFSampleRateHeader`.  This is because many
EDV specifications share common header values, and so their functions and
derived properties should be shared as well.  Moreover, header parsers can be
appended to one another, which saves repetitious coding because the first four
words of any VDIF header are the same.  Indeed, we can create the same header
as above by subclassing :mod:`~baseband.vdif.header.VDIFBaseHeader`::

    >>> class VDIFHeader4Enhanced(vdif.header.VDIFBaseHeader):
    ...     _edv = 42
    ...
    ...     _header_parser = vdif.header.VDIFBaseHeader._header_parser +\
    ...                      vlbi.header.HeaderParser((
    ...                             ('valid_mask_length', (4, 16, 8, 0)),
    ...                             ('sync_pattern', (5, 0, 32, 0xACABFEED)),
    ...                             ('valid_mask_high', (6, 0, 32, 0)),
    ...                             ('valid_mask_low', (7, 0, 32, 0))))
    ...
    ...     def verify(self):
    ...         """Basic checks of header integrity.
    ...         """
    ...         super(VDIFHeader4Enhanced, self).verify()
    ...         assert self['valid_mask_length'] <= 64
    ...
    ...     @property
    ...     def validity_mask(self):
    ...         """64-bit validity mask.
    ...         """
    ...         return (self['valid_mask_high'] << 32) | \
    ...                self['valid_mask_low']
    ...
    ...     @validity_mask.setter
    ...     def validity_mask(self, valid_mask):
    ...         self['valid_mask_high'] = valid_mask >> 32
    ...         self['valid_mask_low'] = valid_mask & (2**31 - 1)

Why did we set ``edv = 42``?  :class:`~!baseband.vdif.header.VDIFHeader`'s
metaclass is designed to prevent accidental overwriting of existing
entries in :obj:`~!baseband.vdif.header.VDIF_HEADER_CLASSES`.  If and doing
so would have returned the exception:

    ``ValueError: EDV 4 already registered in VDIF_HEADER_CLASSES``

We shall see how to override header classes in the next section.  Except for
the EDV, ``VDIFHeader4Enhanced``'s header structure is identical
to ``VDIFHeader4``.  It also contains a few extra functions to enhance the
header's usability.

The ``verify`` function is an optional function that runs upon header
initialization to check its veracity.  Ours simply checks that the
validity mask length is less than 64, but we also call the same function
in the superclass (:class:`~baseband.vdif.header.VDIFBaseHeader`), which
checks that the header is not in 4-word "legacy mode", that the header's
EDV matches that read from the words, that there are eight words, and
that the sync pattern matches ``0xACABFEED``.

``valid_mask_high`` and ``valid_mask_low`` combine to form the validity
mask.  We thus implement a derived property that generates this mask,
and its corresponding setter in cases where the user needs to modify it.

Let's test this enhanced header::

    >>> myenhancedheader = vdif.header.VDIFHeader.fromvalues(
    ...     edv=42, seconds=14363767, nchan=1,
    ...     station=65532, bps=2, complex_data=False,
    ...     thread_id=3, valid_mask_length=64,
    ...     valid_mask_high=(1 << 31),
    ...     valid_mask_low=(1 << 28))
    >>> isinstance(myenhancedheader, VDIFHeader4Enhanced)
    True
    >>> myenhancedheader['valid_mask_high'] == myheader['valid_mask_high']
    True
    >>> myenhancedheader['valid_mask_low'] == myheader['valid_mask_low']
    True
    >>> myenhancedheader.validity_mask == (2**31 << 32) + 2**28
    True
    >>> myenhancedheader.validity_mask = 0b1111
    >>> myenhancedheader['valid_mask_high'] == 0
    True
    >>> myenhancedheader['valid_mask_low'] == 15
    True

.. note::

    If you have implemented support for a new EDV that is widely used, we
    encourage you to incorporated into the Baseband code and submit a pull
    request to Baseband's `GitHub repository 
    <https://github.com/mhvk/baseband>`_, as well as to `register it
    <http://www.vlbi.org/vdif/>`_ (if it is not already registered) with the
    VDIF consortium!

.. _new_edv_replacement:

Replacing an Existing EDV
=========================

In the previous section we mentioned that :class:`~!baseband.vdif.header.VDIFHeader`'s
metaclass is designed to prevent accidental overwriting of existing
entries in :obj:`~!baseband.vdif.header.VDIF_HEADER_CLASSES`, so attempting
to assign two header classes to the same EDV results in an exception.  There
are situations such the one above, however, where we'd like to replace
one header with another.

To get :class:`~!baseband.vdif.header.VDIFHeader` to use ``VDIFHeader4Enhanced``
when ``edv = 4``, we must manually edit the dictionary::

    >>> vdif.header.VDIF_HEADER_CLASSES[4] = VDIFHeader4Enhanced

And then modify the ``_edv`` attribute in ``VDIFHeader4Enhanced``::

    >>> VDIFHeader4Enhanced._edv = 42

:class:`~!baseband.vdif.header.VDIFHeader` will now return instances of
``VDIFHeader4Enhanced`` when reading headers with ``edv = 4``::

    >>> myheader = vdif.header.VDIFHeader.fromvalues(
    ...     edv=4, seconds=14363767, nchan=1,
    ...     station=65532, bps=2, complex_data=False,
    ...     thread_id=3, valid_mask_length=64,
    ...     valid_mask_high=(1 << 31),
    ...     valid_mask_low=(1 << 28))
    >>> isinstance(myheader, VDIFHeader4Enhanced)
    True

.. note::

    Failing to modify ``_edv`` in the class definition will lead to an
    EDV mismatch when ``verify`` is called during header initialization.

This can also be used to override :class:`~!baseband.vdif.header.VDIFHeader`'s
behavior *even for EDVs that are natively supported by Baseband*, which may
prove useful when reading data with corrupted or mislabeled headers.  To
illustrate this, we attempt to read in a corrupted VDIF file originally
from the Dominion Radio Astrophysical Observatory.  This file can be
imported from the baseband data directory::

    >>> from baseband.data import SAMPLE_DRAO_CORRUPT

Naively opening the file with

    ``fh = vdif.open(SAMPLE_DRAO_CORRUPT, 'rs')``

will lead to an AssertionError.  This is because while the headers of the
file purport to be EDV = 0, it deviates from that EDV standard by storing
"link" and "slot" parameters in word 3, byte 3 instead of the thread ID, and an
"eud2" parameter in word 5.  The former indicates the data aquisition
computer node that wrote the data to disk - equivalent to a thread ID -
while the latter indicates data taken over the same time segment.  Meanwhile,
the frame number is meaningless, and the bits-per-sample code is incorrect
(it should be 3 rather than 4 since a one-bit sample has a bits-per-sample
code of 0).

To accommodate these changes, we design an alternate header.  We first
pop the EDV = 0 entry from :obj:`~!baseband.vdif.header.VDIF_HEADER_CLASSES`::

    >>> vdif.header.VDIF_HEADER_CLASSES.pop(0)
    <class 'baseband.vdif.header.VDIFHeader0'>

We then define a replacement class::

    >>> class DRAOVDIFHeader(vdif.header.VDIFHeader0):
    ...     """DRAO VDIF Header
    ... 
    ...     An extension of EDV=0 which uses the thread_id to store link
    ...     and slot numbers, and adds a user keyword (illegal in EDV0,
    ...     but whatever) that identifies data taken at the same time.
    ... 
    ...     The header also corrects 'bits_per_sample' to be properly bps-1.
    ...     """
    ...
    ...     _header_parser = vdif.header.VDIFHeader0._header_parser + \
    ...         vlbi.header.HeaderParser((('link', (3, 16, 4)),
    ...                                   ('slot', (3, 20, 6)),
    ...                                   ('eud2', (5, 0, 32))))
    ... 
    ...     def verify(self):
    ...         pass
    ... 
    ...     @classmethod
    ...     def fromfile(cls, fh, edv=0, verify=False):
    ...         self = super(DRAOVDIFHeader, cls).fromfile(fh, edv=0, 
    ...                                                    verify=False)
    ...         # Correct wrong bps
    ...         self.mutable = True
    ...         self['bits_per_sample'] = 3
    ...         return self

We override ``verify`` because :class:`~!baseband.vdif.header.VDIFHeader0`'s
``verify`` function checks that word 5 contains no data.  We also override
the ``fromfile`` class method such that the ``bits_per_sample`` property
is reset to its proper value whenever a header is read from file.

We can now read in the corrupt file.  We can do this either by manually
reading in the header, then the payload::

    >>> fh = vdif.open(SAMPLE_DRAO_CORRUPT, 'rb')
    >>> header0 = DRAOVDIFHeader.fromfile(fh)
    >>> header0['eud2'] == 667235140
    True
    >>> header0['link'] == 2
    True
    >>> payload0 = vdif.payload.VDIFPayload.fromfile(fh, header0)
    >>> payload0.shape == (header0.samples_per_frame, header0.nchan)
    True

or by modifying :class:`~!baseband.vdif.frame.VDIFFrame` such that
``VDIFFrame._header_class = DRAOVDIFHeader`` before using :func:`~!baseband.vdif.open`.
This is so that header files are read using ``DRAOVDIFHeader.fromfile``
rather than :meth:`~!baseband.vdif.frame.VDIFFrame.fromfile`.

An alternate solution that is compatible with :class:`~!baseband.vdif.base.VDIFStreamReader`
without hacking :class:`~!baseband.vdif.frame.VDIFFrame`, but assumes these
headers are only ever instantiated from file, involves modifying the
bits-per-sample code within ``__init__()``.  Let's remove our previous custom
class, then define a replacement::

    >>> vdif.header.VDIF_HEADER_CLASSES.pop(0)
    <class '__main__.DRAOVDIFHeader'>

::

    >>> class DRAOVDIFHeaderEnhanced(vdif.header.VDIFHeader0):
    ...     """DRAO VDIF Header
    ... 
    ...     An extension of EDV=0 which uses the thread_id to store link and slot
    ...     numbers, and adds a user keyword (illegal in EDV0, but whatever) that
    ...     identifies data taken at the same time.
    ... 
    ...     The header also corrects 'bits_per_sample' to be properly bps-1.
    ...     """
    ...     _header_parser = vdif.header.VDIFHeader0._header_parser + \
    ...         vlbi.header.HeaderParser((('link', (3, 16, 4)),
    ...                                   ('slot', (3, 20, 6)),
    ...                                   ('eud2', (5, 0, 32))))
    ... 
    ...     def __init__(self, words, edv=None, verify=True, **kwargs):
    ...         super(DRAOVDIFHeaderEnhanced, self).__init__(
    ...                 words, verify=False, **kwargs)
    ...         self.mutable = True
    ...         self['bits_per_sample'] = 3
    ...         if verify:
    ...             self.verify()
    ...
    ...     def verify(self):
    ...         pass

We can then use the stream reader without further modification::

    >>> fh2 = vdif.open(SAMPLE_DRAO_CORRUPT, 'rs', frames_per_second=390625)
    >>> fh2.header0['eud2'] == header0['eud2']
    True
    >>> np.array_equal(fh2.read(1), payload0[0])
    True

