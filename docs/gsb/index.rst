.. _gsb:

.. include:: ../tutorials/glossary_substitutions.rst

***
GSB
***

The GMRT software backend (GSB) file format is the standard output of
the initial correlator of the `Giant Metrewave Radio Telescope (GMRT)
<http://www.gmrt.ncra.tifr.res.in/>`_.  The GSB design is described by Roy et
al. (`2010, Exper. Astron. 28:25-60 <https://doi.org/10.1007%2Fs10686-010-9187-0>`_)
with further specifications and operating procedures given on the relevant
`GMRT/GSB pages <http://gmrt.ncra.tifr.res.in/gmrt_hpage/sub_system/gmrt_gsb/index.htm>`_.

.. _gsb_file_structure:

File Structure
==============

A GSB dataset consists of an ASCII file with a sequence of |headers|,
and one or more accompanying binary data files.  Each line in the header and
its corresponding data comprise a :term:`data frame`, though these do not have
explicit divisions in the data files.

Baseband currently supports two forms of GSB data: **rawdump**, for storing
real-valued raw voltage timestreams from individual telescopes, and
**phased**, for storing complex pre-channelized data from the GMRT in phased
array baseband mode.

Data in `rawdump format <http://gmrt.ncra.tifr.res.in/gmrt_hpage/sub_system/
gmrt_gsb/GSB_rawdump_data_format_v2.pdf>`_ is stored in a binary file
representing the voltage stream from one polarization of a single dish.  Each
such file is accompanied by a header file which contains GPS timestamps, in the
form::

    YYYY MM DD HH MM SS 0.SSSSSSSSS

In the default rawdump observing setup for 16 MHz bandwidth, samples are
recorded at a rate of 33.3333... megasamples per second (Msps).  Each sample
is 4 bits in size, and two samples are grouped into bytes such that the oldest
sample occupies the least significant bit.  Each frame consists of **4
megabytes** of data, or :math:`2^{23}` samples; as such, the timespan of one
frame is exactly **0.25165824 s**.

In phased array baseband mode using 16 or 32 MHz bandwidth, the "raw", or
pre-channelized, :term:`sample rate` is either 33.3333... Msps or
66.6666... Msps, with 8 bits per sample.  Channelization via fast Fourier
transform sets the channelized :term:`complete sample` rate to the raw rate
divided by :math:`2N_\mathrm{F}`, where :math:`N_\mathrm{F}` is the number of
Fourier channels (usually 512). The timespan of one frame is **0.25165824 s**,
and the total amount of data associated with a frame is thus either 4 or 8
megabytes.

The data are normally spread over multiple binary files and one accompanying
header file.  The binary files are split by polarization, with for each
polarization either a single file or a pair of files, with the pair containing
the first and second half of the data of each frame.  In each file, the data
associated with a frame is always **4 megabytes**.

The `phased header's structure
<http://gmrt.ncra.tifr.res.in/gmrt_hpage/sub_system/gmrt_gsb/GSB_beam_timestamp_note_v1.pdf>`_
is::

    <PC TIME> <GPS TIME> <SEQ NUMBER> <MEM BLOCK>

where ``<PC TIME>`` and ``<GPS TIME>`` are the less accurate computer-based
and exact GPS-based timestamps, respectively, with the same format as the
rawdump timestamp; ``<SEQ NUMBER>`` is the frame number; and ``<MEM BLOCK>``
a redundant modulo-8 shared memory block number.

.. _gsb_usage:

Usage Notes
===========

This section covers reading and writing GSB files with Baseband; general usage
is covered in the :ref:`Using Baseband <using_baseband>` section.  While
Baseband features the general `baseband.open` and `baseband.file_info`
functions, these cannot read GSB binary files without the accompanying
timestamp file (at which point it is obvious the files are GSB).
`baseband.file_info`, however, can be used on the timestamp file to determine
if it is in rawdump or phased format.

The examples below use the samplefiles in the ``baseband/data/gsb/`` directory,
and the `numpy`, `astropy.units` and `baseband.gsb` modules::

    >>> import numpy as np
    >>> import astropy.units as u
    >>> from baseband import gsb
    >>> from baseband.data import (
    ...     SAMPLE_GSB_RAWDUMP, SAMPLE_GSB_RAWDUMP_HEADER,
    ...     SAMPLE_GSB_PHASED, SAMPLE_GSB_PHASED_HEADER)

A single timestamp file can be opened with `~baseband.gsb.open` in text
mode::

    >>> ft = gsb.open(SAMPLE_GSB_RAWDUMP_HEADER, 'rt')
    >>> ft.read_timestamp()
    <GSBRawdumpHeader gps: 2015 04 27 18 45 00 0.000000240>
    >>> ft.close()

Reading payloads requires the samples per frame or sample rate.  For phased the
sample rate is::

    sample_rate = raw_sample_rate / (2 * nchan)

where the raw sample rate is the pre-channelized one, and ``nchan`` the number
of Fourier channels.  The samples per frame for both rawdump and phased is::

    samples_per_frame = timespan_of_frame * sample_rate

.. note::
    Since the number of samples per frame is an integer number while
    both the frame timespan and sample rate are not, it is better to separately
    caculate ``samples_per_frame`` rather than multiplying
    ``timespan_of_frame`` with ``sample_rate`` in order to avoid rounding
    issues.

Alternatively, if the size of the frame buffer and the frame rate are known, the
former can be used to determine ``samples_per_frame``, and the latter used to
determine ``sample_rate`` by inverting the above equation.

If ``samples_per_frame`` is not given, Baseband assumes the standard payload
size of 4 megabytes, which implies 4 megabytes of data per frame for rawdump,
and 4 or 8 megabytes for phased (depending on the number of files per
polarization).  If ``sample_rate`` is not given, it is calculated from
``samples_per_frame`` assuming that each frame spans 0.25165824 s. (see
:ref:`File Structure <gsb_file_structure>` above).

A single payload file can be opened with `~baseband.gsb.open` in binary mode.
Here, for our sample file, we have to take into account that in order to keep
these files small, their sample size has been reduced to only **4 or 8
kilobytes** worth of samples per frame (for the default timespan).  So, we
define their sample rate here, and use that to calculate ``payload_nbytes``,
the size of one frame in bytes.  Since rawdump samples are 4 bits,
``payload_nbytes`` is just ``samples_per_frame / 2``::

    >>> rawdump_samples_per_frame = 2**13
    >>> payload_nbytes = rawdump_samples_per_frame // 2
    >>> fb = gsb.open(SAMPLE_GSB_RAWDUMP, 'rb', payload_nbytes=payload_nbytes,
    ...               nchan=1, bps=4, complex_data=False)
    >>> payload = fb.read_payload()
    >>> payload[:4]
    array([[ 0.],
           [-2.],
           [-2.],
           [ 0.]], dtype=float32)
    >>> fb.close()

(``payload_nbytes`` for phased data is the size of one frame *divided by the
number of binary files* for each polarization.)

Working in Stream Mode
----------------------

Opening in stream mode allows timestamp and binary files to be read in
concert to create data frames, and also wraps the low-level routines such that
reading and writing is in units of samples, and provides access to header
information.

When opening a rawdump file in stream mode, we pass the timestamp file as the
first argument, and the binary file to the ``raw`` keyword.  As per above, we
also pass ``samples_per_frame``::

    >>> fh_rd = gsb.open(SAMPLE_GSB_RAWDUMP_HEADER, mode='rs',
    ...                  raw=SAMPLE_GSB_RAWDUMP,
    ...                  samples_per_frame=rawdump_samples_per_frame)
    >>> fh_rd.header0
    <GSBRawdumpHeader gps: 2015 04 27 18 45 00 0.000000240>
    >>> dr = fh_rd.read()
    >>> dr.shape
    (81920,)
    >>> dr[:3]
    array([ 0., -2., -2.], dtype=float32)
    >>> fh_rd.close()

To open a phased fileset in stream mode, we package the binary files into a
nested tuple with the format::

    ((L pol stream 1, L pol stream 2), (R pol stream 1, R pol stream 2))

The nested tuple is passed to ``raw``.  Below, we look at the sample files
and open these.  In general, it is a good idea to check that the resulting
stream has the right properties, e.g., the correct bandwidth and the
expected start and stop times. So, we use ``.info`` (note that because our
sample files have been reduced in size we have to pass a non-default
samples per frame and get very small ``sample_rate``, ``bandwidth``, and
``payload_nbytes``)::

    >>> SAMPLE_GSB_PHASED
    (('...sample_gsb_phased.Pol-L1.dat',
      '...sample_gsb_phased.Pol-L2.dat'),
     ('...sample_gsb_phased.Pol-R1.dat',
      '...sample_gsb_phased.Pol-R2.dat'))
    >>> phased_samples_per_frame = 2**3
    >>> fh_ph = gsb.open(SAMPLE_GSB_PHASED_HEADER, mode='rs',
    ...                  raw=SAMPLE_GSB_PHASED,
    ...                  samples_per_frame=phased_samples_per_frame)
    >>> fh_ph.info  # doctest: +FLOAT_CMP
    GSBStream information:
    start_time = 2013-07-27T21:23:55.324108800
    stop_time = 2013-07-27T21:23:57.840691200
    sample_rate = 3.178914388020833e-05 MHz
    shape = (80, 2, 512)
    format = gsb
    bps = 8
    complex_data = True
    verify = True
    bandwidth = 0.016276041666666664 MHz
    n_raw = 2
    payload_nbytes = 4096
    readable = True
    <BLANKLINE>
    checks:  decodable: True
             consistent: True
    <BLANKLINE>
    GSBTimeStampIO information:
    mode = phased
    number_of_frames = 10
    frame_rate = 3.9736429849163546 Hz
    >>> header0 = fh_ph.header0     # To be used for writing, below.
    >>> dp = fh_ph.read()
    >>> dp.shape
    (80, 2, 512)
    >>> dp[0, 0, :3]    # doctest: +FLOAT_CMP
    array([30.+12.j, -1. +8.j,  7.+19.j], dtype=complex64)
    >>> fh_ph.close()

To set up a file for writing, we need to pass names for both
timestamp and raw files, as well as ``sample_rate``, ``samples_per_frame``, and
either the first header or a ``time`` object.  We first calculate
``sample_rate``::

    >>> timespan = 0.25165824 * u.s
    >>> rawdump_sample_rate = (rawdump_samples_per_frame / timespan).to(u.MHz)
    >>> phased_sample_rate = (phased_samples_per_frame / timespan).to(u.MHz)

To write a rawdump file::

    >>> from astropy.time import Time
    >>> fw_rd = gsb.open('test_rawdump.timestamp',
    ...                  mode='ws', raw='test_rawdump.dat',
    ...                  sample_rate=rawdump_sample_rate,
    ...                  samples_per_frame=rawdump_samples_per_frame,
    ...                  time=Time('2015-04-27T13:15:00'))
    >>> fw_rd.write(dr)
    >>> fw_rd.close()
    >>> fh_rd = gsb.open('test_rawdump.timestamp', mode='rs',
    ...                  raw='test_rawdump.dat',
    ...                  sample_rate=rawdump_sample_rate,
    ...                  samples_per_frame=rawdump_samples_per_frame)
    >>> np.all(dr == fh_rd.read())
    True
    >>> fh_rd.close()

To write a phased file, we need to pass a nested tuple of filenames or
filehandles::

    >>> test_phased_bin = (('test_phased_pL1.dat', 'test_phased_pL2.dat'),
    ...                    ('test_phased_pR1.dat', 'test_phased_pR2.dat'))
    >>> fw_ph = gsb.open('test_phased.timestamp',
    ...                  mode='ws', raw=test_phased_bin,
    ...                  sample_rate=phased_sample_rate,
    ...                  samples_per_frame=phased_samples_per_frame,
    ...                  header0=header0)
    >>> fw_ph.write(dp)
    >>> fw_ph.close()
    >>> fh_ph = gsb.open('test_phased.timestamp', mode='rs',
    ...                  raw=test_phased_bin,
    ...                  sample_rate=phased_sample_rate,
    ...                  samples_per_frame=phased_samples_per_frame)
    >>> np.all(dp == fh_ph.read())
    True
    >>> fh_ph.close()

Baseband does not use the PC time in the phased header, and, when writing,
simply uses the same time for both GPS and PC times.  Since the PC time can
drift from the GPS time by several tens of milliseconds,
``test_phased.timestamp`` will not be identical to ``SAMPLE_GSB_PHASED``, even
though we have written the exact same data to file.

.. _gsb_api:

Reference/API
=============

.. automodapi:: baseband.gsb
.. automodapi:: baseband.gsb.header
.. automodapi:: baseband.gsb.payload
.. automodapi:: baseband.gsb.frame
.. automodapi:: baseband.gsb.base
