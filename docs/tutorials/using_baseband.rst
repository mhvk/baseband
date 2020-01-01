.. _using_baseband:

.. include:: ../tutorials/glossary_substitutions.rst

**************
Using Baseband
**************

For most file formats, one can simply import baseband and use `baseband.open`
to access the file.  This gives one a filehandle from which one can read
decoded samples::

    >>> import baseband
    >>> from baseband.data import SAMPLE_DADA
    >>> fh = baseband.open(SAMPLE_DADA)
    >>> fh.read(3)
    array([[ -38.-38.j,  -38.-38.j],
           [ -38.-38.j,  -40. +0.j],
           [-105.+60.j,   85.-15.j]], dtype=complex64)
    >>> fh.close()

For other file formats, a bit more information is needed.  Below, we cover the
basics of :ref:`inspecting files <using_baseband_inspecting>`, :ref:`reading
<using_baseband_reading>` from and :ref:`writing <using_baseband_writing>`
to files, :ref:`converting <using_baseband_converting>` from one format
to another, and :ref:`diagnosing problems <using_baseband_problems>`.
We assume that Baseband as well as `NumPy <https://www.numpy.org/>`_ and the
`Astropy`_ units module have been imported::

    >>> import baseband
    >>> import numpy as np
    >>> import astropy.units as u

.. _using_baseband_inspecting:

Inspecting Files
================

Baseband allows you to quickly determine basic properties of a file, including
what format it is, using the `baseband.file_info` function. For instance, it
shows that the sample VDIF file that comes with Baseband is very short (sample
files can all be found in the `baseband.data` module)::

    >>> import baseband.data
    >>> baseband.file_info(baseband.data.SAMPLE_VDIF)
    Stream information:
    start_time = 2014-06-16T05:56:07.000000000
    stop_time = 2014-06-16T05:56:07.001250000
    sample_rate = 32.0 MHz
    shape = (40000, 8)
    format = vdif
    bps = 2
    complex_data = False
    verify = fix
    readable = True
    <BLANKLINE>
    checks:  decodable: True
             continuous: no obvious gaps
    <BLANKLINE>
    File information:
    edv = 3
    number_of_frames = 16
    thread_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    number_of_framesets = 2
    frame_rate = 1600.0 Hz
    samples_per_frame = 20000
    sample_shape = (8, 1)

The same function will also tell you when more information is needed. For
instance, for Mark 5B files one needs the number of channels used, as well as
(roughly) when the data were taken::

    >>> baseband.file_info(baseband.data.SAMPLE_MARK5B)
    File information:
    format = mark5b
    number_of_frames = 4
    frame_rate = 6400.0 Hz
    bps = 2
    complex_data = False
    readable = False
    <BLANKLINE>
    missing:  nchan: needed to determine sample shape, frame rate, decode data.
              kday, ref_time: needed to infer full times.

    >>> from astropy.time import Time
    >>> baseband.file_info(baseband.data.SAMPLE_MARK5B, nchan=8, ref_time=Time('2014-01-01'))
    Stream information:
    start_time = 2014-06-13T05:30:01.000000000
    stop_time = 2014-06-13T05:30:01.000625000
    sample_rate = 32.0 MHz
    shape = (20000, 8)
    format = mark5b
    bps = 2
    complex_data = False
    verify = fix
    readable = True
    <BLANKLINE>
    checks:  decodable: True
             continuous: no obvious gaps
    <BLANKLINE>
    File information:
    number_of_frames = 4
    frame_rate = 6400.0 Hz
    samples_per_frame = 5000
    sample_shape = (8,)

The information is gleaned from ``info`` properties on the various file and
stream readers (see below).

.. note:: The one format for which ``file_info`` works a bit differently is
   `GSB <gsb>`_, as this format requires separate time-stamp and raw data
   files. Only the timestamp file can be inspected usefully.

.. _using_baseband_reading:

Reading Files
=============

Opening Files
-------------

As shown at the very start, files can be opened with the general
`baseband.open` function. This will try to determine the file type using
`~baseband.file_info`, load the corresponding baseband module, and then open
the file using that module's master input/output function.

Generally, if one knows the file type, one might as well work with the
corresponding module directly.  For instance, to explicitly use the DADA
reader to open the sample DADA file included in Baseband, one can use the DADA
module's `~baseband.dada.open` function::

    >>> from baseband import dada
    >>> from baseband.data import SAMPLE_DADA
    >>> fh = dada.open(SAMPLE_DADA, 'rs')
    >>> fh.read(3)
    array([[ -38.-38.j,  -38.-38.j],
           [ -38.-38.j,  -40. +0.j],
           [-105.+60.j,   85.-15.j]], dtype=complex64)
    >>> fh.close()

In general, file I/O and data manipulation use the same syntax across all file
formats.  When opening Mark 4 and Mark 5B files, however, some additional
arguments may need to be passed (as was the case above for inspecting a Mark
5B file, and indeed this is a good way to find out what is needed).
Notes on such features and quirks of individual formats can be
found in the API entries of their ``open`` functions, and within the
:ref:`Specific file format <specific_file_formats_toc>` documentation.

For the rest of this section, we will stick to VDIF files.

Decoding Data and the Sample File Pointer
-----------------------------------------

By giving the openers a ``'rs'`` flag, which is the default, we open files in
"stream reader" mode, where a file is accessed as if it were a stream of
samples.  For VDIF, `~baseband.vdif.open` will then return an instance of
`~baseband.vdif.base.VDIFStreamReader`, which wraps a raw data file with
methods to decode the binary |data frames| and seek to and read data
|samples|.  To decode the first 12 samples into a `~numpy.ndarray`, we would
use the `~baseband.vdif.base.VDIFStreamReader.read` method::

    >>> from baseband import vdif
    >>> from baseband.data import SAMPLE_VDIF
    >>> fh = vdif.open(SAMPLE_VDIF, 'rs')
    >>> d = fh.read(12)
    >>> type(d)
    <... 'numpy.ndarray'>
    >>> d.shape
    (12, 8)
    >>> d[:, 0].astype(int)    # First thread.
    array([-1, -1,  3, -1,  1, -1,  3, -1,  1,  3, -1,  1])

As discussed in detail in the :ref:`VDIF section <vdif>`, VDIF files are
sequences of data frames, each of which is comprised of a :term:`header` (which
holds information like the time at which the data was taken) and a
:term:`payload`, or block of data.  Multiple concurrent time streams can be
stored within a single frame; each of these is called a ":term:`channel`".
Moreover, groups of channels can be stored over multiple frames, each of which
is called a ":term:`thread`".  Our sample file is an "8-thread, single-channel
file" (8 concurrent time streams with 1 stream per frame), and in the example
above, ``fh.read`` decoded the first 12 samples from all 8 threads, mapping
thread number to the second axis of the decoded data array.  Reading files with
multiple threads and channels will produce 3-dimensional arrays.

``fh`` includes ``shape``, ``size`` and ``ndim``, which give the shape, total
number of elements and dimensionality of the file's entire dataset if it was
decoded into an array.  The number of |complete samples| - the set of samples
from all available threads and channels for one point in time - in the file is
given by the first element in ``shape``::

    >>> fh.shape    # Shape of all data from the file in decoded array form.
    (40000, 8)
    >>> fh.shape[0] # Number of complete samples.
    40000
    >>> fh.size
    320000
    >>> fh.ndim
    2

The shape of a single complete sample, including names indicating the meaning
of shape dimensions, is retrievable using::

    >>> fh.sample_shape
    SampleShape(nthread=8)

By default, dimensions of length unity are |squeezed|, or removed from the
sample shape.  To retain them, we can pass ``squeeze=False`` to
`~baseband.vdif.open`::

    >>> fhns = vdif.open(SAMPLE_VDIF, 'rs', squeeze=False)
    >>> fhns.sample_shape    # Sample shape now keeps channel dimension.
    SampleShape(nthread=8, nchan=1)
    >>> fhns.ndim            # fh.shape and fh.ndim also change with squeezing.
    3
    >>> d2 = fhns.read(12)
    >>> d2.shape             # Decoded data has channel dimension.
    (12, 8, 1)
    >>> fhns.close()

Basic information about the file is obtained by either by ``fh.info`` or simply
``fh`` itself::

    >>> fh.info
    Stream information:
    start_time = 2014-06-16T05:56:07.000000000
    stop_time = 2014-06-16T05:56:07.001250000
    sample_rate = 32.0 MHz
    shape = (40000, 8)
    format = vdif
    bps = 2
    complex_data = False
    verify = fix
    readable = True
    <BLANKLINE>
    checks:  decodable: True
             continuous: no obvious gaps
    <BLANKLINE>
    File information:
    edv = 3
    number_of_frames = 16
    thread_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    number_of_framesets = 2
    frame_rate = 1600.0 Hz
    samples_per_frame = 20000
    sample_shape = (8, 1)

    >>> fh
    <VDIFStreamReader name=... offset=12
        sample_rate=32.0 MHz, samples_per_frame=20000,
        sample_shape=SampleShape(nthread=8),
        bps=2, complex_data=False, edv=3, station=65532,
        start_time=2014-06-16T05:56:07.000000000>

Not coincidentally, the first is identical to what we :ref:`found above
<using_baseband_inspecting>` using `~baseband.file_info`.

The filehandle itself also shows the ``offset``, the current location of the
sample file pointer. Above, it is at ``12`` since we have read in 12 (complete)
samples.  If we called ``fh.read (12)`` again we would get the next 12 samples.
If we instead called ``fh.read()``, it would read from the pointer's *current*
position to the end of the file.  If we wanted all the data in one array, we
would move the file pointer back to the start of file, using ``fh.seek``,
before reading::

    >>> fh.seek(0)      # Seek to sample 0.  Seek returns its offset in counts.
    0
    >>> d_complete = fh.read()
    >>> d_complete.shape
    (40000, 8)

We can also move the pointer with respect to the end of file by passing ``2``
as a second argument::

    >>> fh.seek(-100, 2)    # Second arg is 0 (start of file) by default.
    39900
    >>> d_end = fh.read(100)
    >>> np.array_equal(d_complete[-100:], d_end)
    True

``-100`` means 100 samples before the end of file, so ``d_end`` is equal to
the last 100 entries of ``d_complete``.  Baseband only keeps the most recently
accessed data frame in memory, making it possible to analyze (normally large)
files through selective decoding using ``seek`` and ``read``.

.. note::

    As with file pointers in general, ``fh.seek`` will not return an error if
    one seeks beyond the end of file.  Attempting to read beyond
    the end of file, however, will result in an ``EOFError``.

To determine where the pointer is located, we use ``fh.tell()``::

    >>> fh.tell()
    40000
    >>> fh.close()

Caution should be used when decoding large blocks of data using ``fh.read``.
For typical files, the resulting arrays are far too large to hold in memory.

Seeking and Telling in Time With the Sample Pointer
---------------------------------------------------

We can use ``seek`` and ``tell`` with units of time rather than samples.  To do
this with ``tell``, we can pass an appropriate `astropy.units.Unit` object to
its optional ``unit`` parameter::

    >>> fh = vdif.open(SAMPLE_VDIF, 'rs')
    >>> fh.seek(40000)
    40000
    >>> fh.tell(unit=u.ms)
    <Quantity 1.25 ms>

Passing the string ``'time'`` reports the pointer's location in absolute time::

    >>> fh.tell(unit='time')
    <Time object: scale='utc' format='isot' value=2014-06-16T05:56:07.001250000>

We can also pass an absolute `astropy.time.Time`, or a positive or negative time
difference `~astropy.time.TimeDelta` or `astropy.units.Quantity` to ``seek``.
If the offset is a `~astropy.time.Time` object, the second argument to seek is
ignored.::

    >>> from astropy.time.core import TimeDelta
    >>> from astropy.time import Time
    >>> fh.seek(TimeDelta(-5e-4, format='sec'), 2)  # Seek -0.5 ms from end.
    24000
    >>> fh.seek(0.25*u.ms, 1)  # Seek 0.25 ms from current position.
    32000
    >>> # Seek to specific time.
    >>> fh.seek(Time('2014-06-16T05:56:07.001125'))
    36000

We can retrieve the time of the first sample in the file using ``start_time``,
the time immediately after the last sample using ``stop_time``, and the time
of the pointer's current location (equivalent to ``fh.tell(unit='time')``)
using ``time``::

    >>> fh.start_time
    <Time object: scale='utc' format='isot' value=2014-06-16T05:56:07.000000000>
    >>> fh.stop_time
    <Time object: scale='utc' format='isot' value=2014-06-16T05:56:07.001250000>
    >>> fh.time
    <Time object: scale='utc' format='isot' value=2014-06-16T05:56:07.001125000>
    >>> fh.close()

Extracting Header Information
-----------------------------

The first header of the file is stored as the ``header0`` attribute of the
stream reader object; it gives direct access to header properties via keyword
lookup::

    >>> with vdif.open(SAMPLE_VDIF, 'rs') as fh:
    ...     header0 = fh.header0
    >>> header0['frame_length']
    629

The full list of keywords is available by printing out ``header0``::

    >>> header0
    <VDIFHeader3 invalid_data: False,
                 legacy_mode: False,
                 seconds: 14363767,
                 _1_30_2: 0,
                 ref_epoch: 28,
                 frame_nr: 0,
                 vdif_version: 1,
                 lg2_nchan: 0,
                 frame_length: 629,
                 complex_data: False,
                 bits_per_sample: 1,
                 thread_id: 1,
                 station_id: 65532,
                 edv: 3,
                 sampling_unit: True,
                 sampling_rate: 16,
                 sync_pattern: 0xacabfeed,
                 loif_tuning: 859832320,
                 _7_28_4: 15,
                 dbe_unit: 2,
                 if_nr: 0,
                 subband: 1,
                 sideband: True,
                 major_rev: 1,
                 minor_rev: 5,
                 personality: 131>

A number of derived properties, such as the time (as a `~astropy.time.Time`
object), are also available through the header object::

    >>> header0.time
    <Time object: scale='utc' format='isot' value=2014-06-16T05:56:07.000000000>

These are listed in the API for each header class.  For example, the sample
VDIF file's headers are of class::

    >>> type(header0)
    <class 'baseband.vdif.header.VDIFHeader3'>

and so its attributes can be found `here <baseband.vdif.header.VDIFHeader3>`.

Reading Specific Components of the Data
---------------------------------------

By default, ``fh.read()`` returns complete samples, i.e. with all
available threads, polarizations or channels. If we were only interested in
decoding a :term:`subset` of the complete sample, we can select specific
components by passing indexing objects to the ``subset`` keyword in open.  For
example, if we only wanted thread 3 of the sample VDIF file::

    >>> fh = vdif.open(SAMPLE_VDIF, 'rs', subset=3)
    >>> fh.sample_shape
    ()
    >>> d = fh.read(20000)
    >>> d.shape
    (20000,)
    >>> fh.subset
    (3,)
    >>> fh.close()

Since by default data are squeezed, one obtains a data stream with just a
single dimension.  If one would like to keep all information, one has to pass
``squeeze=False`` and also make ``subset`` a list (or slice)::

    >>> fh = vdif.open(SAMPLE_VDIF, 'rs', subset=[3], squeeze=False)
    >>> fh.sample_shape
    SampleShape(nthread=1, nchan=1)
    >>> d = fh.read(20000)
    >>> d.shape
    (20000, 1, 1)
    >>> fh.close()

Data with multi-dimensional samples can be subset by passing a `tuple` of
indexing objects with the same dimensional ordering as the (possibly squeezed)
sample shape; in the case of the sample VDIF with ``squeeze=False``, this is
threads, then channels. For example, if we wished to select threads 1 and 3,
and channel 0::

    >>> fh = vdif.open(SAMPLE_VDIF, 'rs', subset=([1, 3], 0), squeeze=False)
    >>> fh.sample_shape
    SampleShape(nthread=2)
    >>> fh.close()

Generally, ``subset`` accepts any object that can be used to `index
<https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html>`_ a
`numpy.ndarray`, including advanced indexing (as done above, with
``subset=([1, 3], 0)``).  If possible, slices should be used instead
of list of integers, since indexing with them returns a view rather
than a copy and thus avoid unnecessary processing and memory allocation.
(An exception to this is VDIF threads, where the subset is used to selectively
read specific threads, and thus is not used for actual slicing of the data.)

.. _using_baseband_writing:

Writing to Files and Format Conversion
======================================

Writing to a File
-----------------

To write data to disk, we again use ``open``.  Writing data in a particular
format requires both the header and data samples.  For modifying an existing
file, we have both the old header and old data handy.

As a simple example, let's read in the 8-thread, single-channel sample VDIF
file and rewrite it as an single-thread, 8-channel one, which, for example, may
be necessary for compatibility with `DSPSR
<https://github.com/demorest/dspsr>`_::

    >>> import baseband.vdif as vdif
    >>> from baseband.data import SAMPLE_VDIF
    >>> fr = vdif.open(SAMPLE_VDIF, 'rs')
    >>> fw = vdif.open('test_vdif.vdif', 'ws',
    ...                sample_rate=fr.sample_rate,
    ...                samples_per_frame=fr.samples_per_frame // 8,
    ...                nthread=1, nchan=fr.sample_shape.nthread,
    ...                complex_data=fr.complex_data, bps=fr.bps,
    ...                edv=fr.header0.edv, station=fr.header0.station,
    ...                time=fr.start_time)

The minimal parameters needed to generate a file are listed under the
documentation for each format's ``open``, though comprehensive lists can be
found in the documentation for each format's stream writer class (eg. for
VDIF, it's under `~baseband.vdif.base.VDIFStreamWriter`).  In practice we
specify as many relevant header properties as available to obtain a particular
file structure.  If we possess the *exact* first header of the file, it can
simply be passed to ``open`` via the ``header`` keyword.  In the example above,
though, we manually switch the values of ``nthread`` and ``nchan``.  Because
VDIF EDV = 3 requires each frame's payload to contain 5000 bytes, and ``nchan``
is now a factor of 8 larger, we decrease ``samples_per_frame``, the number of
complete (i.e. all threads and channels included) samples per frame, by a
factor of 8.

Encoding samples and writing data to file is done by passing data arrays into
``fw``'s `~baseband.vdif.base.VDIFStreamWriter.write` method.  The first
dimension of the arrays is sample number, and the remaining dimensions must be
as given by ``fw.sample_shape``::

    >>> fw.sample_shape
    SampleShape(nchan=8)

In this case, the required dimensions are the same as the arrays from
``fr.read``.  We can thus write the data to file using::

    >>> while fr.tell() < fr.shape[0]:
    ...     fw.write(fr.read(fr.samples_per_frame))
    >>> fr.close()
    >>> fw.close()

For our sample file, we could simply have written

    ``fw.write(fr.read())``

instead of the loop, but for large files, reading and writing should be done in
smaller chunks to minimize memory usage.  Baseband stores only the data frame
or frame set being read or written to in memory.

We can check the validity of our new file by re-opening it::

    >>> fr = vdif.open(SAMPLE_VDIF, 'rs')
    >>> fh = vdif.open('test_vdif.vdif', 'rs')
    >>> fh.sample_shape
    SampleShape(nchan=8)
    >>> np.all(fr.read() == fh.read())
    True
    >>> fr.close()
    >>> fh.close()

.. note:: One can also use the top-level `~baseband.open` function for writing,
          with the file format passed in via its ``format`` argument.

.. _using_baseband_converting:

File Format Conversion
----------------------

It is often preferable to convert data from one file format to another that
offers wider compatibility, or better fits the structure of the data.  As an
example, we convert the sample Mark 4 data to VDIF.

Since we don't have a VDIF header handy, we pass the relevant Mark 4 header
values into `vdif.open <baseband.vdif.open>` to create one::

    >>> import baseband.mark4 as mark4
    >>> from baseband.data import SAMPLE_MARK4
    >>> fr = mark4.open(SAMPLE_MARK4, 'rs', ntrack=64, decade=2010)
    >>> spf = 640       # fanout * 160 = 640 invalid samples per Mark 4 frame
    >>> fw = vdif.open('m4convert.vdif', 'ws', sample_rate=fr.sample_rate,
    ...                samples_per_frame=spf, nthread=1,
    ...                nchan=fr.sample_shape.nchan,
    ...                complex_data=fr.complex_data, bps=fr.bps,
    ...                edv=1, time=fr.start_time)

We choose ``edv = 1`` since it's the simplest VDIF EDV whose header includes a
sampling rate. The concept of threads does not exist in Mark 4, so the file
effectively has ``nthread = 1``.  As discussed in the :ref:`Mark 4
documentation <mark4>`, the data at the start of each frame is effectively
overwritten by the header and are represented by invalid samples in the stream
reader.  We set ``samples_per_frame`` to ``640`` so that each section of
invalid data is captured in a single frame.

We now write the data to file, manually flagging each invalid data frame::

    >>> while fr.tell() < fr.shape[0]:
    ...     d = fr.read(fr.samples_per_frame)
    ...     fw.write(d[:640], valid=False)
    ...     fw.write(d[640:])
    >>> fr.close()
    >>> fw.close()

Lastly, we check our new file::

    >>> fr = mark4.open(SAMPLE_MARK4, 'rs', ntrack=64, decade=2010)
    >>> fh = vdif.open('m4convert.vdif', 'rs')
    >>> np.all(fr.read() == fh.read())
    True
    >>> fr.close()
    >>> fh.close()

For file format conversion in general, we have to consider how to properly
scale our data to make the best use of the dynamic range of the new encoded
format. For VLBI formats like VDIF, Mark 4 and Mark 5B, samples of the same
size have the same scale, which is why we did not have to rescale our data when
writing 2-bits-per-sample Mark 4 data to a 2-bits-per-sample VDIF file.
Rescaling is necessary, though, to convert DADA or GSB to VDIF.  For examples
of rescaling, see the ``baseband/tests/test_conversion.py`` file.

.. _using_baseband_multifile:

Reading or Writing to a Sequence of Files
=========================================

Data from one continuous observation is sometimes spread over a sequence of
files. Baseband includes the `~baseband.helpers.sequentialfile` module for
reading in a sequence as if it were one contiguous file.  This module is called
when a list, tuple or filename template is passed to eg. `baseband.open` or
`baseband.vdif.open`, making the syntax for handling multiple files nearly
identical to that for single ones.

As an example, we write the data from the sample VDIF file
``baseband/data/sample.vdif`` into a sequence of two files and then read the
files back in.  We first load the required data::

    >>> from baseband import vdif
    >>> from baseband.data import SAMPLE_VDIF
    >>> import numpy as np
    >>> fh = vdif.open(SAMPLE_VDIF, 'rs')
    >>> d = fh.read()

We then create a sequence of filenames::

    >>> filenames = ["seqvdif_{0}".format(i) for i in range(2)]

When passing ``filenames`` to `~baseband.vdif.open`, we must also pass
``file_size``, the file size in bytes, in addition to the usual ``kwargs`` for
writing a file.  Since we wish to split the sample file in two, and the file
consists of two framesets, we set ``file_size`` to the byte size of one
frameset (we could have equivalently set it to ``fh.fh_raw.seek(0, 2) // 2``)::

    >>> file_size = 8 * fh.header0.frame_nbytes
    >>> fw = vdif.open(filenames, 'ws', header0=fh.header0,
    ...                file_size=file_size, sample_rate=fh.sample_rate,
    ...                nthread=fh.sample_shape.nthread)
    >>> fw.write(d)
    >>> fw.close()    # This implicitly closes fwr.

.. note::

    ``file_size`` sets the maximum size a file can reach before the
    writer writes to the next one, so setting ``file_size`` to a larger
    value than above will lead to the two files having different sizes.  By
    default, ``file_size=None``, meaning it can be arbitrarily large, in which
    case only one file will be created.

We now read the sequence and confirm their contents are identical to those of
the sample file::

    >>> fr = vdif.open(filenames, 'rs', sample_rate=fh.sample_rate)
    >>> fr.header0.time == fh.header0.time
    True
    >>> np.all(fr.read() == d)
    True
    >>> fr.close()

When reading, the filename sequence **must be ordered in time**.

We can also open the second file on its own and confirm it contains the second
frameset of the sample file::

    >>> fsf = vdif.open(filenames[1], mode='rs', sample_rate=fh.sample_rate)
    >>> fh.seek(fh.shape[0] // 2)    # Seek to start of second frameset.
    20000
    >>> fsf.header0.time == fh.time
    True
    >>> np.all(fsf.read() == fh.read())
    True
    >>> fsf.close()

In situations where the ``file_size`` is known, but not the total number of
files to write, one may use the `~baseband.helpers.sequentialfile.FileNameSequencer`
class to create an iterable without a user-defined size.  The class is
initialized with a template string that can be formatted with keywords, and a
optional ``header`` that can either be an actual header or a `dict` with the
relevant keywords.  The template may also contain the special keyword
'{file_nr}', which is equal to the indexing value (instead of a header entry).

As an example, let us create a sequencer::

    >>> from baseband.helpers import sequentialfile as sf
    >>> filenames = sf.FileNameSequencer('f.edv{edv:d}.{file_nr:03d}.vdif',
    ...                                  header=fh.header0)

Indexing the sequencer using square brackets returns a filename::

    >>> filenames[0]
    'f.edv3.000.vdif'
    >>> filenames[42]
    'f.edv3.042.vdif'

The sequencer has extracted the EDV from the header we passed in, and the
file number from the index.  We can use the sequencer to write a VDIF file
sequence::

    >>> fw = vdif.open(filenames, 'ws', header0=fh.header0,
    ...                file_size=file_size, sample_rate=fh.sample_rate,
    ...                nthread=fh.sample_shape.nthread)
    >>> d = np.concatenate([d, d, d])
    >>> fw.write(d)
    >>> fw.close()

This creates 6 files::

    >>> import glob
    >>> len(glob.glob("f.edv*.vdif"))
    6

We can read the file sequence using the same sequencer.  In reading mode, the
sequencer determines the number of files by finding the largest file available
that fits the template::

    >>> fr = vdif.open(filenames, 'rs', sample_rate=fh.sample_rate)
    >>> fr.header0.time == fh.header0.time
    True
    >>> np.all(fr.read() == d)
    True
    >>> fr.close()
    >>> fh.close()  # Close sample file as well.

Because DADA and GUPPI data are usually stored in file sequences with names
derived from header values - eg. 'puppi_58132_J1810+1744_2176.0010.raw',
their format openers have template support built-in.  For usage details, please
see the API entries for `baseband.dada.open` and `baseband.guppi.open`.

.. _using_baseband_problems:

Diagnosing problems with baseband files
=======================================

Little is more annoying than starting a very long analysis script only to find
the reader crashed with an error near the end.  Unfortunately, while there is
only one way for success, there are many for failure.  Some, though, can be
found by :ref:`inspecting files <using_baseband_inspecting>`.  To see
what would show up for a file that misses a frame, we first construct one::

    >>> from astropy.time import Time
    >>> from baseband import vdif
    >>> fc = vdif.open('corrupt.vdif', 'ws', edv=1, nthread=2,
    ...                bps=8, samples_per_frame=16,
    ...                time=Time('J2010'), sample_rate=16*u.kHz)
    >>> fc.write(np.zeros((8000, 2)))
    >>> fc.fh_raw.seek(-100, 1)
    47900
    >>> fc.write(np.zeros((8000, 2)))
    >>> fc.close()

Here, rewinding the internal raw file pointer a bit to simulate "missing
bytes" is an implementation detail that one should not rely on!

Now check its `~baseband.vdif.base.VDIFStreamReader.info`::

    >>> fh = baseband.vdif.open('corrupt.vdif', 'rs', verify=True)
    >>> fh.info.readable
    False
    >>> fh.info
    Stream information:
    start_time = 2009-12-31T23:58:53.816000000
    stop_time = 2009-12-31T23:58:54.816000000
    sample_rate = 0.016 MHz
    shape = (16000, 2)
    format = vdif
    bps = 8
    complex_data = False
    verify = True
    readable = False
    <BLANKLINE>
    checks:  decodable: True
             continuous: False
    <BLANKLINE>
    errors:  continuous: While reading at 7968: AssertionError()
    <BLANKLINE>
    warnings:  number_of_frames: file contains non-integer number (1997.9166666666667) of frames
    <BLANKLINE>
    File information:
    edv = 1
    thread_ids = [0, 1]
    frame_rate = 1000.0 Hz
    samples_per_frame = 16
    sample_shape = (2, 1)
    >>> fh.close()

In detail, the error is given for a position earlier than the one we
corrupted, because internally baseband reads a frame ahead since a
corrupted frame typically means something is bad before as well.

This particular problem is not bad, since the VDIF reader can deal with
missing frames.  Indeed, when one opens the file with the default
``verify='fix'``, one gets::

    >>> fh = baseband.vdif.open('corrupt.vdif', 'rs')
    >>> fh.info
    Stream information:
    start_time = 2009-12-31T23:58:53.816000000
    stop_time = 2009-12-31T23:58:54.816000000
    sample_rate = 0.016 MHz
    shape = (16000, 2)
    format = vdif
    bps = 8
    complex_data = False
    verify = fix
    readable = True
    <BLANKLINE>
    checks:  decodable: True
             continuous: fixable gaps
    <BLANKLINE>
    warnings:  number_of_frames: file contains non-integer number (1997.9166666666667) of frames
               continuous: While reading at 7968: problem loading frame set 498. Thread(s) [1] missing; set to invalid.
    <BLANKLINE>
    File information:
    edv = 1
    thread_ids = [0, 1]
    frame_rate = 1000.0 Hz
    samples_per_frame = 16
    sample_shape = (2, 1)
    >>> fh.close()
