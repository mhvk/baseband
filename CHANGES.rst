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
  `baseband.vlbi_base.encoding.encode_4bit_base`. [#250]

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
