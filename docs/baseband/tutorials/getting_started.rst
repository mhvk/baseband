.. _getting_started:

***************
Getting Started
***************

This tutorial covers the basic features of Baseband.  It assumes that Numpy
has been imported::

    >>> import numpy as np

.. _getting_started_reading:

Reading Files
=============

Opening Files
-------------

Each format supported by Baseband has a master input/output function,
accessible by importing the corresponding format module. For example, to read
the sample VDIF file located in ``baseband/data/sample.vdif``::

	>>> import baseband.vdif as vdif
	>>> from baseband.data import SAMPLE_VDIF
	>>> fh = vdif.open(SAMPLE_VDIF, 'rs')

To close the file::

    >>> fh.close()

The same syntax can be used to open a file of any supported format.  To open
Baseband's sample DADA file, for example::

    >>> from baseband import dada
    >>> from baseband.data import SAMPLE_DADA
    >>> fh_dada = dada.open(SAMPLE_DADA, 'rs')
    >>> fh_dada.close()

In general, file I/O and data manipulation use the same syntax across all
file formats.  When using ``open`` for Mark 4 and Mark 5B files, however, two
keywords - ``ntrack``, and ``decade`` - may need to be set manually.  For these
and VDIF, ``frames_per_second`` may also need to be passed if it can't be read
or inferred from the file. Notes on such features and quirks of individual
file formats can be found within the :ref:`Specific file format
<specific_file_formats_toc>` section.

For now, let's go back to using VDIF files.

Decoding Data and the Sample File Pointer
-----------------------------------------

We gave `~baseband.vdif.open` the ``rs`` flag, which opens the file in "stream
reader" mode.  The function thus returns an instance of
`~baseband.vdif.base.VDIFStreamReader`, which is a wrapper around the standard
`io.BufferedReader` binary file reader that decodes files as a stream of
data frames and adds methods to seek to and read individual data samples.  To
read the first 12 data samples into an `~numpy.ndarray`, we would use the
``read`` method::

    >>> fh = vdif.open(SAMPLE_VDIF, 'rs')
	>>> d = fh.read(12)
	>>> type(d)
	<class 'numpy.ndarray'>
	>>> d.shape
	(12, 8)
	>>> d[:, 0].astype(int)  # First thread.
	array([-1, -1,  3, -1,  1, -1,  3, -1,  1,  3, -1,  1])

The sample VDIF file has 8 concurrent frequency bands, or "channels", which are
mapped as the columns of the array.  We can access information from the header
by printing ``fh``::

    >>> fh
    <VDIFStreamReader name=... offset=12
        nthread=8, samples_per_frame=20000, nchan=1,
        frames_per_second=1600, complex_data=False, bps=2, edv=3,
        station=65532, (start) time=2014-06-16T05:56:07.000000000>

.. note::

	VDIF files will very likely contain multiple data frames, and therefore
	muptile headers, per file.  The information printed by the stream reader
	comes only from the first header in the file.  (Other file formats, such as
	DADA, generally have a single header per file, making this distinction
	unimportant for them.)

The ``offset``, above, gives the current location of the sample file
pointer - it's at ``12`` since we have just read in 12 samples.  If we called
``fh.read(12)`` again we would get the next 12 samples.  If we called 
``fh.read()``, it would read from the point's current position to the end of
file.  If we wanted all the data in one array, this would be annoying, since we
would then need to append ``d``, above.  Instead, we can move the file pointer
back to the start of file, using ``fh.seek``, before reading::

	>>> fh.seek(0)		# Seek to sample 0.  Seek reports its offset in counts.
	0
	>>> d_complete = fh.read()
	>>> d_complete.shape
	(40000, 8)

.. note::

	``fh.read()`` returns a **copy** of the data from ``fh``.

We can also move the pointer with respect to the end of file by passing ``2``
as a second argument (as with `io.BufferedReader` pointers)::

	>>> fh.seek(-100, 2)	# Second arg is 0 (start of file) by default.
	39900
	>>> d_end = fh.read(100)
	>>> np.array_equal(d_complete[-100:], d_end)
	True

Here, ``-100`` means 100 counts before the end of file, so ``d_end`` is equal to
the last 100 entries of ``d_complete``.  Baseband only keeps the most recently
accessed data frame in memory, so selective decoding using ``seek`` and
``read`` is useful when examining extremely large files.

To determine where the pointer is located, we use ``fh.tell()``::

	>>> fh.tell()
	40000
    >>> fh.close()

Seeking and Telling in Time With the Sample Pointer
---------------------------------------------------

We can use ``seek`` and ``tell`` with units of time, rather than samples.  To do
this with ``tell``, we can pass an appropriate `astropy.units.Unit` object to
its optional ``unit`` parameter::

    >>> import astropy.units as u
    >>> fh = vdif.open(SAMPLE_VDIF, 'rs')
    >>> fh.seek(40000)
    40000
    >>> fh.tell(unit=u.ms)
    <Quantity 1.25 ms>

Passing the special string ``time`` reports the pointer's location in absolute
time::

    >>> fh.tell(unit='time')
    <Time object: scale='utc' format='isot' value=2014-06-16T05:56:07.001250000>

We can also pass an absolute `astropy.time.Time`, or a positive or negative time
difference `~astropy.time.TimeDelta` or `astropy.units.Quantity` to ``seek``. 
If the offset is a `~!astropy.time.Time` object, the second argument to seek is
ignored.

::

    >>> from astropy.time.core import TimeDelta
    >>> from astropy.time import Time
    >>> fh.seek(TimeDelta(-5e-4, format='sec'), 2)  # Seek -0.5 ms from end.
    24000
    >>> fh.seek(0.25*u.ms, 1)  # Seek 0.25 ms from current position.
    32000
    >>> # Seek to time index 2014/06/16 5:56:07.001125
    >>> fh.seek(Time('2014-06-16T05:56:07.001125', precision=6))
    36000
	>>> fh.close()

Extracting Header Information
-----------------------------

The first header of the file is stored as the ``header0`` attribute of the
stream reader object, which gives direct access to header properties via keyword
lookup::

    >>> with vdif.open(SAMPLE_VDIF, 'rs') as fh:
    ...     header0 = fh.header0
    >>> header0['frame_length']
    629

The full list of keywords is available through the ``keys`` method::

    >>> header0.keys()
    odict_keys(['invalid_data', 'legacy_mode', ...])

A number of derived properties, such as the time, are also
available through the header object.  

    >>> header0.time
    <Time object: scale='utc' format='isot' value=2014-06-16T05:56:07.000000000>

These are listed in the API under each header class's attributes.  For example,
the sample VDIF file's headers are of class::

    >>> type(header0)
    <class 'baseband.vdif.header.VDIFHeader3'>

and so its attributes can be found in its `API entry
<baseband.vdif.header.VDIFHeader3>`.

Opening Specific Threads/Channels From Files
--------------------------------------------

In general, files can contain multiple channels of an observation, and for VDIF
in particular different channels can be bundled into "threads".  If we were
only interested in specific threads/channels, we can select them using the
``thread_ids`` keyword::

    >>> fh = vdif.open(SAMPLE_VDIF, 'rs', thread_ids=[2, 3])
    >>> d = fh.read(20000)
    >>> d.shape
    (20000, 2)
    >>> fh.close()

For VDIF, this selects the specified threads (each of which many have multiple
channels), while for others this selects the specified channels.


.. _getting_started_writing:

Writing to Files
================

To write data to disk, we again use the master ``open``.  Writing data in a
particular format requires both the header and data samples.  For modifying an
existing file, we have the old header as well as the old data handy.

As a simple example, let's read in the single-channel, 8-threaded sample VDIF
file and rewrite it as an 8-channel, single-thread one, which for example, may
be necessary for compatibility with certain data reduction codes::

    >>> fh = vdif.open(SAMPLE_VDIF, 'rs')
    >>> fho = vdif.open('test_vdif.vdif', 'ws',
    ...                 nthread=fh.nchan, nchan=fh.nthread,
    ...                 frames_per_second=fh.frames_per_second,
    ...                 samples_per_frame=fh.samples_per_frame // 8,
    ...                 complex_data=fh.complex_data,
    ...                 bps=fh.bps, edv=fh.header0.edv,
    ...                 station=fh.header0.station, time=fh.time0)

The minimal parameters needed to generate a file are listed under the
documentation for each file format's ``open``, though the comprehensive
list can be found under that for each's format's stream writer class (eg. for
VDIF, it is under `~baseband.vdif.base.VDIFStreamWriter`).  In practice we
specify as many relevant header properties as available to obtain a particular
file structure.  If we possess the *exact* first header of the file, it can
simply be passed to ``open`` via the ``header`` keyword.  In the example above,
though, we manually switch the ``nthread`` and ``nchan``.  Because VDIF EDV = 3
requires each frame's payload contain 5000 bytes, and ``nchan`` is a factor of 8
larger, we decrease ``samples_per_frame``, the number of complete (i.e. all
channels included) samples per frame, by a factor of 8.

Writing the data to file,

::

    >>> data = fh.read()
    >>> fho.write(data)     # In this case, no need to reshape data.
    >>> fh.close()
    >>> fho.close()

In the case of large files, reading and writing should be done in small chunks
to minimize memory usage.  Baseband stores only the data frame or frame set
being read or written to in memory.

We can check the validity of our new file by re-opening it::

    >>> fh = vdif.open(SAMPLE_VDIF, 'rs')
    >>> fho = vdif.open('test_vdif.vdif', 'rs')
    >>> fho.nchan
    8
    >>> fho.nthread
    1
    >>> np.all(fh.read() == fho.read())
    True
    >>> fh.close()
    >>> fho.close()

we pass header values (or )


To do the sameFor example, the sample
Mark 4 file's data is divided into two data frames.  To save the first
frame as a separate file, we first read it into memory::

    >>> import baseband.mark4 as mark4
    >>> from baseband.data import SAMPLE_MARK4
    >>> fr = mark4.open(SAMPLE_MARK4, 'rb')  # Open in binary reader mode.
    >>> fr.find_frame(64)  # Find first frame.
    2696
    >>> f0 = fr.read_frame(64, 2010)
    >>> fr.close()

We use ``open`` in binary reader mode, which reads in the file and gives
access to frame reading methods, but does not set up the sample file
pointer.  In addition, Mark 4 files don't need to start at a frame boundary, so
``fr.find_frame`` is used to discover the first one.

We then write it to a new file.  To open a file in write mode, one generally
needs a filename or binary file object in write mode, and a sample header with
the correct initial time.  Writing Mark 4 data also requires we provide
``ntrack`` and ``decade``, just like with reading, as well as the frame rate,
since this cannot be inferred by scanning a file that doesn't yet exist::

    >>> fw = mark4.open('sample_mark4_segment.m4', 'ws', header=f0.header,
    ...                 ntrack=64, decade=2010, frames_per_second=400)
    >>> fw.write(f0.data)
    >>> fw.close()

We can re-open the file to check that its data frame is identical to ``f0``::

    >>> fwr = mark4.open('sample_mark4_segment.m4', 'rb')
    >>> fwr.find_frame(64)
    0
    >>> assert fwr.read_frame(64, 2010) == f0
    >>> fwr.close()

Specifics on writing individual file formats, including necessary additional
parameters, can be found in the API documentation for each file format's
``open`` function.  Seeking to and picking out frames is most easily done using
the binary (rather than sample) file pointer; the one in ``fr`` is accessible as
``fr.fh_raw``.  More on this pointer, and on reading and writing data frames as
we have done above, can be found in :ref:`Reading and Writing Data Frames
<frame_io>`.

We could attempt to write only a few samples to a file while using the same
header, but this will produce a warning:

    ``UserWarning: Closing with partial buffer remaining.  Writing padded frame,
    marked as invalid.``

This is because the data frame is much larger than the number of samples we've
written to it.  The Mark 4 specification, however, requires at least 4960
samples per channel (with a fan-out ratio of 4) in a frame, so padding is
inevitable when writing only a handful of values.

File Format Conversion
======================

An alternative solution is to write the samples to VDIF.  Since we don't
have a VDIF header handy, we pass the relevant Mark 4 header values into
``vdif.open`` in order to create one.  Let's write out the first 1920 samples::

    >>> from baseband import vdif
    >>> import astropy.units as u
    >>> fr = mark4.open(SAMPLE_MARK4, 'rs', ntrack=64, decade=2010)
    >>> spf = 640  # fanout * 160 = 640 invalid samples per Mark 4 frame
    >>> f_rate = (fr.frames_per_second * fr.samples_per_frame / spf)*u.Hz
    >>> fw = vdif.open('m4convert.vdif', 'ws', edv=1, nthread=1,
    ...                samples_per_frame=spf, nchan=fr.nchan,
    ...                framerate=f_rate, complex_data=fr.complex_data, 
    ...                bps=fr.bps, time=fr.time0)
    >>> d = fr.read(1920)
    >>> fw.write(d[:640], invalid_data=True)
    >>> fw.write(d[640:])
    >>> fr.close()
    >>> fw.close()

There are some format-specific arguments that we have to manually set. We
choose ``edv = 1`` since it is the simplest VDIF EDV whose header includes a
frame rate (see the :ref:`documentation on VDIF <vdif>`). The concept of threads
does not exist in Mark 4, so we set ``nthread = 1`` to keep the data the same
shape when read out using ``vdif.open``.  As discussed in the :ref:`Mark 4
documentation <mark4>`, the data at the start of each frame overwritten by the
header is represented by invalid samples in the stream reader.  We set
``samples_per_frame`` to ``640`` so that each section of invalid data is
captured in a single frame.  Only one such section exists in our data,
and we manually flag it as invalid.  The framerate is naturally set to 50 kHz
once we set the ``samples_per_frame``.

Lastly, we check that we can read back the data::

    >>> fr = vdif.open('m4convert.vdif', 'rs')
    >>> d2 = fr.read()
    >>> np.array_equal(d, d2)
    True
    >>> fr.close()