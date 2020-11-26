4.0.3 (2020-11-26)
==================

Bug Fixes
---------

- Ensure that ``pathlib.Path`` objects are recognized as valid in the various
  openers. [#467]

- Raise a proper ``FileNotFoundError`` instead of an obscure ``AttributeError``
  if trying to get ``file_info`` on a non-existing file. [#467]

- Pass on all "irrelevant" arguments not understood by ``file_info`` to the
  general opener, so they can be used or raise ``TypeError`` in a place where
  it will be clearer why that happens. [#468]

- Support for VDIF EDV3 data with payload size of 1000 bytes. [#456]


4.0.2 (2020-10-23)
==================

Bug Fixes
---------

- Fix the GUPPIHeader class incorrectly ignoring the STT_OFFS header
  keyword. [#457]

4.0.1 (2020-07-31)
==================

Bug Fixes
---------

- Allow the GUPPI reader to assume channel-first ordering by default, i.e.,
  no longer insist that PKTFMT is one of '1SFA' or 'SIMPLE'. Instead, ``info``
  will include a warning for formats not known to work. [#453]

4.0 (2020-07-18)
================

- The minimum versions required by baseband are now python 3.7, numpy 1.17
  and astropy 4.0.

- Baseband now requires the (very small) ``entrypoints`` package.

New Features
------------

- Baseband now provides an ``baseband.io`` entry point, which allows other
  packages to make new readers accessible to baseband by defining an entry
  point in their ``setup.cfg``. [#418]

- Similarly, baseband also provides an ``baseband.tasks`` entry point, which
  allows other packages to define tasks useful for processing baseband
  data by defining an entry point in their ``setup.cfg``. This is primarily
  intended for the future ``baseband-tasks`` package. [#445]

API Changes
-----------

The internals of baseband have undergone fairly substantial refactoring to
make the classes more coherent. This should not affect users directly, but may
affect those that have built their own readers.

- Following python 3.9, ``HeaderParser`` instances (which are subclasses of
  ``dict``), can now be merged together using the ``|`` operator. For
  backward compatibility, using the ``+`` operator will remain supported.
  [#424]

- All ``StreamWriters`` now require an explicit ``header0`` to be passed
  in (as was already the case for DADA and GUPPI). Creation of a ``header0``
  from keyword arguments is now done inside the opener. [#417]

- The ``vlbi_base`` module has been deprecated in favour of ``base``,
  and ``VLBI`` prefixes of classes have been removed where these were
  not specific to actual VLBI data, leaving only ``VLBIHeaderBase``,
  ``VLBIFileReaderBase``, and ``VLBIStreamReaderBase``.  [#425]

- The stream base classes will now try to get information that is not
  passed in explicitly from ``header0``. Given this change, the keyword
  argument ``unsliced_shape`` become somewhat illogical, so was changed
  to ``sample_shape`` (still referring to the pre-squeeze and subset
  shape) [#415, #433]

- Support for memory mapping of payloads has been moved into the base
  ``PayloadBase`` and ``FrameBase`` classes and thus is available for all
  formats. [#427]

- Payloads and frames now all take ``sample_shape`` as an argument, instead
  of some taking ``nchan``. [#429]

Bug Fixes
---------

- Extraneous arguments to stream writers are no longer ignored, but give
  rise to a ``TypeError``. [#417]

- The GUPPI stream reader now will include any overlap samples from the
  last frame. [#431]

Other Changes and Additions
---------------------------

- All baseband formats now support passing in template strings for stream
  readers and writers (e.g., ``'{file_nr:07d}.vdif'``). [#417]

- The headers for VDIF and Mark 4 now expose standard ``complex_data``
  and ``sample_shape`` properties, to match what is done for the other
  headers. Mark 5B headers expose only ``complex_data``, as the sample
  shape cannot be inferred from the header. [#414, #428]

- General classes to help writing ``open`` and ``info`` functions are now
  provided in ``baseband.vlbi_base.FileOpener`` and ``FileInfo``. [#418]

- The general ``open`` and ``file_open`` functions are now defined in
  ``baseband.io`` (but still imported at the top level). They are able
  to use any format defined via the plugin system. [#444]

3.2.1 (2020-06-24)
==================

Bug Fixes
---------

- For GSB phased data, fix the interpretation of ``sample_rate`` in
  calculating ``payload_nbytes``. [#410]

- Fix pickling for GSB phased data.

3.2 (2020-06-11)
================

New Features
------------

- All file and stream readers can now be pickled.  Writers still cannot,
  since those do not allow appending. [#395]

Bug Fixes
---------

- Mark 4 data written with the non-standard channel assignment used at Ft
  can now be read and written. [#380]

- For GSB phased data, the default ``payload_nbytes`` has now been corrected
  so that it is always 4 MiB. [#401]

- For GSB phased data, the ``sample_rate`` argument is now correctly
  interpreted as the rate of complete samples (previously, the number of
  channels were ignored). [#401]

Other Changes and Additions
---------------------------

- The ``temporary_offset`` context manager of file readers now allows to
  pass in a possible initial offset to go to. [#390]

- The GSB stream reader ``.info`` has been updated to include a consistency
  check of the size of the raw files with the number of frames inferred
  from the timestamp file. [#407]

3.1.1 (2020-04-05)
==================

Bug Fixes
---------

- Mark 5B is fixed so that writing files is now also possible on big-endian
  architectures.


3.1 (2020-01-23)
================

Bug Fixes
---------

- Frame rates are now calculated correctly also for Mark 4 data in which the
  first frame is the last within a second. [#341]

- Fixed a bug where a VDIF header was not found correctly if the file pointer
  was very close to the start of a header already. [#346]

- In VDIF header verification, include that the implied payload must have
  non-negative size. [#348]

- Mark 4 now checks by default (``verify=True``) that frames are ordered
  correctly. [#349]

- ``find_header`` will now always check that the frame corresponding to
  a header is complete (i.e., fits within the file). [#354]

- The ``count`` argument to ``.read()`` no longer is changed in-place, making
  it safe to pass in array scalars or dimensionless quantities. [#373]

Other Changes and Additions
---------------------------

- The Mark 4, Mark 5B, and VDIF stream readers are now able to replace
  missing pieces of files with zeros using ``verify='fix'``. This is
  also the new default; use ``verify=True`` for the old behaviour of
  raising an error on any inconsistency. [#357]

- The ``VDIFFileReader`` gained a new ``get_thread_ids()`` method, which
  will scan through frames to determine the threads present in the file.
  This is now used inside ``VDIFStreamReader`` and, combined with the above,
  allows reading of files that have missing threads in their first frame
  set. [#361]

- The stream reader info now also checks whether streams are continuous
  by reading the first and last sample, allowing a simple way to check
  whether the file will likely pose problems before possibly spending
  a lot of time reading it. [#364]

- Much faster localization of Mark 5B frames. [#351]

- VLBI file readers have gained a new method ``locate_frames`` that finds
  frame starts near the current location. [#354]

- For VLBI file readers, ``find_header`` now raises an exception if no
  frame is found (rather than return `None`).

- The Mark 4 file reader's ``locate_frame`` has been deprecated. Its
  functionality is replaced by ``locate_frames`` and ``find_header``. [#354]

- Custom stream readers can now override only part of reading a given frame
  and testing that it is the right one. [#355]

- The ``HeaderParser`` class was refactored and simplified, making setting
  keys faster. [#356]

- ``info`` now also provides the number of frames in a file. [#364]


3.0 (2019-08-28)
================

- This version only supports python3.

New Features
------------

- File information now includes whether a file can be read and decoded.
  The ``readable()`` method on stream readers also includes whether the
  data in a file can be decoded. [#316]

Bug Fixes
---------

- Empty GUPPI headers can now be created without having to pass in
  ``verify=False``. This is needed for astropy 3.2, which initializes an empty
  header in its revamped ``.fromstring`` method. [#314]

- VDIF multichannel headers and payloads are now forced to have power-of-two
  bits per sample. [#315]

- Bits per complete sample for VDIF payloads are now calculated correctly also
  for non power-of-two bits per sample. [#315]

- Guppi raw file info now presents the correct sample rate, corrected for
  overlap. [#319]

- All headers now check that ``samples_per_frame`` are set to possible numbers.
  [#325]

- Getting ``.info`` on closed files no longer leads to an error (though
  no information can be retrieved). [#326]

Other Changes and Additions
---------------------------

- Increased speed of VDIF stream reading by removing redundant verification.
  Reduces the overhead for verification for VDIF CHIME data from 50% (factor
  1.5) to 13%. [#321]

2.0 (2018-12-12)
================

- VDIF and Mark 5B readers and writers now support 1 bit per sample.
  [#277, #278]

Bug Fixes
---------

- VDIF reader will now properly ignore corrupt last frames. [#273]

- Mark5B reader more robust against headers not being parsed correctly
  in ``Mark5BFileReader.find_header``. [#275]

- All stream readers now have a proper ``dtype`` attribute, not a
  corresponding ``np.float32`` or ``np.complex64``. [#280]

- GUPPI stream readers no longer emit warnings on not quite FITS compliant
  headers. [#283]

Other Changes and Additions
---------------------------

- Added release procedure to the documentation.  [#268]

1.2 (2018-07-27)
================

New Features
------------

- Expanded support for acccessing sequences of files to VLBI format
  openers and `baseband.open`.  Enabled `baseband.guppi.open` to open file
  sequences using string templates like with `baseband.dada.open`. [#254]

- Created `baseband.helpers.sequentialfile.FileNameSequencer`, a
  general-purpose filename sequencer that can be passed to any format opener.
  [#253]

Other Changes and Additions
---------------------------

- Moved the Getting Started section to :ref:`"Using Baseband"
  <using_baseband>`, and created a new quickstart tutorial under :ref:`Getting
  Started <getting_started>` to better assist new users.  [#260]

1.1.1 (2018-07-24)
==================

Bug Fixes
---------

- Ensure ``gsb`` times can be decoded with astropy-dev (which is to become
  astropy 3.1). [#249]

- Fixed rounding error when encoding 4-bit data using
  ``baseband.vlbi_base.encoding.encode_4bit_base``. [#250]

- Added GUPPI/PUPPI to the list of file formats used by `baseband.open` and
  `baseband.file_info`.  [#251]

1.1 (2018-06-06)
================

New Features
------------

- Added a new `baseband.file_info` function, which can be used to inspect
  data files. [#200]

- Added a general file opener, `baseband.open` which for a set of formats
  will check whether the file is of that format, and then load it using the
  corresponding module. [#198]

- Allow users to pass a ``verify`` keyword to file openers reading streams.
  [#233]

- Added support for the GUPPI format. [#212]

- Enabled `baseband.dada.open` to read streams where the last frame has an
  incomplete payload. [#228]

API Changes
-----------

- In analogy with Mark 5B, VDIF header time getting and setting now requires
  a frame rate rather than a sample rate. [#217, #218]

- DADA and GUPPI now support passing either a ``start_time`` or ``offset``
  (in addition to ``time``) to set the start time in the header. [#240]

Bug Fixes
---------

Other Changes and Additions
---------------------------

- The `baseband.data` module with sample data files now has an explicit entry
  in the documentation. [#198]

- Increased speed of VLBI stream reading by changing the way header sync
  patterns are stored, and removing redundant verification steps.  VDIF
  sequential decode is now 5 - 10% faster (depending on the number of
  threads). [#241]

1.0.1 (2018-06-04)
==================

Bug Fixes
---------

- Fixed a bug in `baseband.dada.open` where passing a ``squeeze`` setting is
  ignored when also passing header keywords in 'ws' mode. [#211]

- Raise an exception rather than return incorrect times for Mark 5B files
  in which the fractional seconds are not set. [#216]

Other Changes and Additions
---------------------------

- Fixed broken links and typos in the documentation. [#211]


1.0.0 (2018-04-09)
==================

- Initial release.
