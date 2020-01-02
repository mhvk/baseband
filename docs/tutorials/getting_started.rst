.. _getting_started:

.. include:: ../tutorials/glossary_substitutions.rst

*****************************
Getting Started with Baseband
*****************************

This quickstart tutorial is meant to help the reader hit the ground running
with Baseband.  For more detail, including writing to files, see :ref:`Using
Baseband <using_baseband>`.

For installation instructions, please see :ref:`Installing Baseband
<install_baseband>`.

When using Baseband, we typically will also use `numpy`, `astropy.units`, and
`astropy.time.Time`. Let's import all of these::

    >>> import baseband
    >>> import numpy as np
    >>> import astropy.units as u
    >>> from astropy.time import Time

.. _getting_started_opening:

Opening Files
=============

For this tutorial, we'll use two sample files::

    >>> from baseband.data import SAMPLE_VDIF, SAMPLE_MARK5B

The first file is a VDIF one created from `EVN <https://www.evlbi.org/>`_/`VLBA
<https://public.nrao.edu/telescopes/vlba/>`_ observations of `Black Widow
pulsar PSR B1957+20 <https://en.wikipedia.org/wiki/Black_Widow_Pulsar>`_,
while the second is a Mark 5B from EVN/`WSRT
<https://www.astron.nl/radio-observatory/public/public-0>`_ observations of the
same pulsar.

To open the VDIF file::

    >>> fh_vdif = baseband.open(SAMPLE_VDIF)

Opening the Mark 5B file is slightly more involved, as not all required
metadata is stored in the file itself::

    >>> fh_m5b = baseband.open(SAMPLE_MARK5B, nchan=8, sample_rate=32*u.MHz,
    ...                        ref_time=Time('2014-06-13 12:00:00'))

Here, we've manually passed in as keywords the number of |channels|, the
:term:`sample rate` (number of samples per channel per second) as an
`astropy.units.Quantity`, and a reference time within 500 days of the start of
the observation as an `astropy.time.Time`.  That last keyword is needed to
properly read timestamps from the Mark 5B file.

`baseband.open` tries to open files using all available formats, returning
whichever is successful.  If you know the format of your file, you can pass
its name with the ``format`` keyword, or directly use its format opener (for
VDIF, it is `baseband.vdif.open`).  Also, the `baseband.file_info` function can
help determine the format and any missing information needed by `baseband.open`
- see :ref:`Inspecting Files <using_baseband_inspecting>`.

Do you have a sequence of files you want to read in?  You can pass a list of
filenames to `baseband.open`, and it will open them up as if they were a single
file!  See :ref:`Reading or Writing to a Sequence of Files <using_baseband_multifile>`.

.. _getting_started_reading:

Reading Files
=============

Radio baseband files are generally composed of blocks of binary data, or
|payloads|, stored alongside corresponding metadata, or |headers|.  Each
header and payload combination is known as a :term:`data frame`, and most
formats feature files composed of a long series of frames.

Baseband file objects are frame-reading wrappers around Python file objects,
and have the same interface, including
`~baseband.vlbi_base.base.VLBIStreamReaderBase.seek`
for seeking to different parts of the file,
`~baseband.vlbi_base.base.VLBIStreamReaderBase.tell` for reporting the file
pointer's current position, and
`~baseband.vlbi_base.base.VLBIStreamReaderBase.read` for reading data.  The
main difference is that Baseband file objects read and navigate in units of
samples.

Let's read some samples from the VDIF file::

    >>> data = fh_vdif.read(3)
    >>> data  # doctest: +FLOAT_CMP
    array([[-1.      ,  1.      ,  1.      , -1.      , -1.      , -1.      ,
             3.316505,  3.316505],
           [-1.      ,  1.      , -1.      ,  1.      ,  1.      ,  1.      ,
             3.316505,  3.316505],
           [ 3.316505,  1.      , -1.      , -1.      ,  1.      ,  3.316505,
            -3.316505,  3.316505]], dtype=float32)
    >>> data.shape
    (3, 8)

Baseband decodes binary data into `~numpy.ndarray` objects.  Notice we
input ``3``, and received an array of shape ``(3, 8)``; this is because
there are 8 VDIF |threads|.  Threads and channels represent different
|components| of the data such as polarizations or frequency sub-bands, and the
collection of all components at one point in time is referred to as a
:term:`complete sample`.  Baseband reads in units of complete samples,
and works with sample rates in units of complete samples per second (including
with the Mark 5B example above). Like an `~numpy.ndarray`, calling
``fh_vdif.shape`` returns the shape of the entire dataset::

    >>> fh_vdif.shape
    (40000, 8)

The first axis represents time, and all additional axes represent the shape of
a complete sample.  A labelled version of the complete sample shape is given
by::

    >>> fh_vdif.sample_shape
    SampleShape(nthread=8)

Baseband extracts basic properties and header metadata from opened files.
Notably, the start and end times of the file are given by::

    >>> fh_vdif.start_time
    <Time object: scale='utc' format='isot' value=2014-06-16T05:56:07.000000000>
    >>> fh_vdif.stop_time
    <Time object: scale='utc' format='isot' value=2014-06-16T05:56:07.001250000>

For an overview of the file, we can either print ``fh_vdif`` itself, or use the
``info`` method::

    >>> fh_vdif
    <VDIFStreamReader name=... offset=3
        sample_rate=32.0 MHz, samples_per_frame=20000,
        sample_shape=SampleShape(nthread=8),
        bps=2, complex_data=False, edv=3, station=65532,
        start_time=2014-06-16T05:56:07.000000000>
    >>> fh_vdif.info
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

Seeking is also done in units of complete samples, which is equivalent to
seeking in timesteps.  Let's move forward 100 complete samples::

    >>> fh_vdif.seek(100)
    100

Seeking from the end or current position is also possible, using the same
syntax as for typical file objects. It is also possible to seek in units of
time::

    >>> fh_vdif.seek(-1000, 2)    # Seek 1000 samples from end.
    39000
    >>> fh_vdif.seek(10*u.us, 1)    # Seek 10 us from current position.
    39320

``fh_vdif.tell`` returns the current offset in samples or in time::

    >>> fh_vdif.tell()
    39320
    >>> fh_vdif.tell(unit=u.us)    # Time since start of file.
    <Quantity 1228.75 us>
    >>> fh_vdif.tell(unit='time')
    <Time object: scale='utc' format='isot' value=2014-06-16T05:56:07.001228750>

Finally, we close both files::

    >>> fh_vdif.close()
    >>> fh_m5b.close()
