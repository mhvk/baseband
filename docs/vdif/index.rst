.. _vdif:

.. include:: ../tutorials/glossary_substitutions.rst

****
VDIF
****

The `VLBI Data Interchange Format (VDIF) <https://www.vlbi.org/vdif/>`_ was
introduced in 2009 to standardize VLBI data transfer and storage.  Detailed
specifications are found in VDIF's `specification document
<https://vlbi.org/wp-content/uploads/2019/03/VDIF_specification_Release_1.1.1.pdf>`_.

.. _vdif_file_structure:

File Structure
==============

A VDIF file is composed of |data frames|.  Each has a :term:`header` of eight
32-bit words (32 bytes; the exception is the "legacy VDIF" format, which is
four words, or 16 bytes, long), and a :term:`payload` that ranges from 32 bytes
to ~134 megabytes.  Both are little-endian.  The first four words of a VDIF
header hold the same information in all VDIF files, but the last four words
hold optional user-defined data.  The layout of these four words is specified
by the file's **extended-data version**, or EDV.  More detailed information on
the header can be found in the :ref:`tutorial for supporting a new VDIF EDV
<new_edv>`.

A data frame may carry one or multiple |channels|, and a :term:`stream` of data
frames all carrying the same (set of) channels is known as a :term:`thread` and
denoted by its thread ID.  The collection of frames representing the same time
segment (and all possible thread IDs) is called a  :term:`data frameset` (or
just "frameset").

Strict time and thread ID ordering of frames in the stream, while considered
part of VDIF best practices, is not mandated, and cannot be guaranteed during
data transmission over the internet.

.. _vdif_usage:

Usage Notes
===========

This section covers reading and writing VDIF files with Baseband; general
usage can be found under the :ref:`Using Baseband <using_baseband>` section.
For situations in which one is unsure of a file's format, Baseband features the
general `baseband.open` and `baseband.file_info` functions, which are also
discussed in :ref:`Using Baseband <using_baseband>`.  The examples below use
the small sample file ``baseband/data/sample.vdif``, and the `numpy`,
`astropy.units`, and `baseband.vdif` modules::

    >>> import numpy as np
    >>> from baseband import vdif
    >>> import astropy.units as u
    >>> from baseband.data import SAMPLE_VDIF

Simple reading and writing of VDIF files can be done entirely using
`~baseband.vdif.open`. Opening in binary mode provides a normal file
reader, but extended with methods to read a `~baseband.vdif.VDIFFrameSet`
data container for storing a frame set as well as
`~baseband.vdif.VDIFFrame` one for storing a single frame::

    >>> fh = vdif.open(SAMPLE_VDIF, 'rb')
    >>> fs = fh.read_frameset()
    >>> fs.data.shape
    (20000, 8, 1)
    >>> fr = fh.read_frame()
    >>> fr.data.shape
    (20000, 1)
    >>> fh.close()

(As with other formats, ``fr.data`` is a read-only property of the frame.)

Opening in stream mode wraps the low-level routines such that reading
and writing is in units of samples.  It also provides access to header
information::

    >>> fh = vdif.open(SAMPLE_VDIF, 'rs')
    >>> fh
    <VDIFStreamReader name=... offset=0
        sample_rate=32.0 MHz, samples_per_frame=20000,
        sample_shape=SampleShape(nthread=8),
        bps=2, complex_data=False, edv=3, station=65532,
        start_time=2014-06-16T05:56:07.000000000>
    >>> d = fh.read(12)
    >>> d.shape
    (12, 8)
    >>> d[:, 0].astype(int)  # first thread
    array([-1, -1,  3, -1,  1, -1,  3, -1,  1,  3, -1,  1])
    >>> fh.close()

To set up a file for writing needs quite a bit of header information. Not
coincidentally, what is given by the reader above suffices::


    >>> from astropy.time import Time
    >>> fw = vdif.open('try.vdif', 'ws', sample_rate=32*u.MHz,
    ...                samples_per_frame=20000, nchan=1, nthread=2,
    ...                complex_data=False, bps=2, edv=3, station=65532,
    ...                time=Time('2014-06-16T05:56:07.000000000'))
    >>> with vdif.open(SAMPLE_VDIF, 'rs', subset=[1, 3]) as fh:
    ...    d = fh.read(20000)  # Get some data to write
    >>> fw.write(d)
    >>> fw.close()
    >>> fh = vdif.open('try.vdif', 'rs')
    >>> d2 = fh.read(12)
    >>> np.all(d[:12] == d2)
    True
    >>> fh.close()

Here is a simple example to copy a VDIF file.  We use the ``sort=False`` option
to ensure the frames are written exactly in the same order, so the files should
be identical::

    >>> with vdif.open(SAMPLE_VDIF, 'rb') as fr, vdif.open('try.vdif', 'wb') as fw:
    ...     while True:
    ...         try:
    ...             fw.write_frameset(fr.read_frameset(sort=False))
    ...         except:
    ...             break

For small files, one could just do::

    >>> with vdif.open(SAMPLE_VDIF, 'rs') as fr, \
    ...         vdif.open('try.vdif', 'ws', header0=fr.header0,
    ...                   sample_rate=fr.sample_rate,
    ...                   nthread=fr.sample_shape.nthread) as fw:
    ...     fw.write(fr.read())

This copies everything to memory, though, and some header information is lost.

.. _vdif_troubleshooting:

Troubleshooting
===============

In situations where the VDIF files being handled are corrupted or modified
in an unusual way, using `~baseband.vdif.open` will likely lead to an
exception being raised or to unexpected behavior.  In such cases, it may still
be possible to read in the data.  Below, we provide a few solutions and
workarounds to do so.

.. note::
    This list is certainly incomplete.   If you have an issue (solved
    or otherwise) you believe should be on this list, please e-mail
    the :ref:`contributors <contributors>`.

AssertionError when checking EDV in header ``verify`` function
--------------------------------------------------------------

All VDIF header classes (other than `~baseband.vdif.header.VDIFLegacyHeader`)
check, using their ``verify`` function, that the EDV read from file matches
the class EDV.  If they do not, the following line

    ``assert self.edv is None or self.edv == self['edv']``

returns an AssertionError.  If this occurs because the VDIF EDV is not yet
supported by Baseband, support can be added by implementing a custom header
class.  If the EDV is supported, but the header deviates from the format
found in the `VLBI.org EDV registry <https://www.vlbi.org/vdif/>`_, the
best solution is to create a custom header class, then override the
subclass selector in `~baseband.vdif.header.VDIFHeader`.  Tutorials
for doing either can be found :ref:`here <new_edv>`.

EOFError encountered in ``_get_frame_rate`` when reading
---------------------------------------------------------

When the sample rate is not input by the user and cannot be deduced from header
information (if EDV = 1 or, the sample rate is found in the header), Baseband
tries to determine the frame rate using the private method ``_get_frame_rate``
in `~baseband.vdif.base.VDIFStreamReader` (and then multiply by the
samples per frame to obtain the sample rate).  This function raises `EOFError`
if the file contains less than one second of data, or is corrupt.  In either
case the file can be opened still by explicitly passing in the sample rate to
`~baseband.vdif.open` via the ``sample_rate`` keyword.

.. _vdif_api:

Reference/API
=============

.. automodapi:: baseband.vdif
.. automodapi:: baseband.vdif.header
   :include-all-objects:
.. automodapi:: baseband.vdif.payload
.. automodapi:: baseband.vdif.frame
.. automodapi:: baseband.vdif.file_info
.. automodapi:: baseband.vdif.base
