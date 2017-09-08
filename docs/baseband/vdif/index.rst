.. _vdif:

****
VDIF
****

The `VLBI Data Interchange Format (VDIF) <http://www.vlbi.org/vdif/>`_ was
introduced in 2009 to standardize VLBI data transfer and storage.  Detailed
specifications are found in VDIF's `specification document
<http://www.vlbi.org/vdif/docs/VDIF_specification_Release_1.1.1.pdf>`_.

.. _vdif_file_structure:

VDIF File Structure
===================

A VDIF file or data transmission is composed of a sequence of **data frames**,
each of which is comprised of a self-identifying data frame
header followed by a sequence, or **payload**, of data covering a single time
segment of observations from one or more frequency sub-bands.  The header
is a pre-defined 32-bytes long, while the payload is task-specific and can
range from 32 bytes to ~134 megabytes.  Both are little-endian and grouped
into 32-bit **words**.  The first four words of a VDIF header hold the same
information in all VDIF files, but the last four words hold optional
user-defined data.  The layout of these four words is specified by the file's
**extended-data version**, or EDV.  More detailed information on the header
can be found in the :ref:`tutorial for supporting a new VDIF EDV <new_edv>`.

A data frame may carry one or multiple frequency sub-bands or Fourier
channels, and we refer to either of these as **channels** for short.  A sequence
of data frames all carrying the same (set of) channels is called a **data
thread**, denoted by its thread ID.  A data set consisting of multiple
concurrent threads is transmitted or stored as a serial sequence of frames
called a **data stream**.  The collection of frames that cover all threads for a
single time segment - equivalently, all thread IDs for the same header time and
frame number - is a **dataframe set** (or just "frame set").

Strict time ordering of frames in the stream, while considered part of VDIF
best practices, is not mandated, and cannot be guaranteed during data
transmission over the internet.

.. _vdif_usage:

Usage Notes
===========

This section covers VDIF-specific features of Baseband.  Tutorials for general
usage can be found under the :ref:`Using Baseband <using_baseband_toc>` section.
The examples below use the small sample file ``baseband/data/sample.vdif``,
and assume the `numpy` and `baseband.vdif` modules have been imported::

    >>> import numpy as np
    >>> from baseband import vdif
    >>> from baseband.data import SAMPLE_VDIF

Baseband defines the :class:`~baseband.vdif.VDIFFrameSet` data container for
storing a frame set as well as :class:`~baseband.vdif.VDIFFrame` one for storing
a single frame. Opening in a VDIF file binary mode provides a file reader
extended with methods to read both::

    >>> fh = vdif.open(SAMPLE_VDIF, 'rb')
    >>> fs = fh.read_frameset()
    >>> fs.data.shape
    (8, 20000, 1)
    >>> fr = fh.read_frame()
    >>> fr.data.shape
    (20000, 1)
    >>> fh.close()

As with other formats, ``fr.data`` is a read-only property of the frame. 
``fs.data``, though, is a lazy property which, when it is first called, decodes
the entire frame set payload into a `numpy.ndarray`.  The values in ``fs.data``
can freely be modified, but are *not* transmitted back to the raw payload data.

To set up a file for writing needs quite a bit of header information. Not
coincidentally, what is given by the reader above suffices::

    >>> fh = vdif.open(SAMPLE_VDIF, 'rs', thread_ids=[2, 3])
    >>> d = fh.read(20000)

::

    >>> from astropy.time import Time
    >>> import astropy.units as u
    >>> fw = vdif.open('try.vdif', 'ws',
    ...                nthread=2, samples_per_frame=20000, nchan=1,
    ...                frames_per_second=1600, complex_data=False, bps=2, edv=3,
    ...                station=65532, time=Time('2014-06-16T05:56:07.000000000'))
    >>> fw.write(d)
    >>> fw.close()
    >>> fh = vdif.open('try.vdif', 'rs')
    >>> d2 = fh.read(12)
    >>> np.all(d[:12] == d2)
    True
    >>> fh.close()

Example to copy a VDIF file.  Here, we use the ``sort=False`` option to ensure
the frames are written exactly in the same order, so the files should be
identical.::

    >>> with vdif.open(SAMPLE_VDIF, 'rb') as fr, vdif.open('try.vdif', 'wb') as fw:
    ...     while True:
    ...         try:
    ...             fw.write_frameset(fr.read_frameset(sort=False))
    ...         except:
    ...             break

For small files, one could just do::

    >>> with vdif.open(SAMPLE_VDIF, 'rs') as fr, vdif.open(
    ...         'try.vdif', 'ws', header=fr.header0, nthread=fr.nthread) as fw:
    ...     fw.write(fr.read())

This copies everything to memory, though, and some header information is lost.

.. _vdif_troubleshooting:

Troubleshooting
===============

In situations where the VDIF files being handled are corrupted or modified
in an unusual way, using :func:`~baseband.vdif.open` will likely lead
either to an exception being raised or to unexpected behavior.  In such
cases, it may still be possible to read in the data.  Below, we provide a
few solutions and workarounds to do so.

.. note::
    This list is certainly incomplete.   If you have an issue (solved
    or otherwise) you believe should be on this list, please e-mail
    the :ref:`contributors <contributors>`.

AssertionError when checking EDV in header ``verify`` function
--------------------------------------------------------------

All VDIF header classes (other than :class:`~baseband.vdif.header.VDIFLegacyHeader`)
check, using their ``verify`` function, that the EDV read from file matches
the class EDV.  If they do not, the following line

    ``assert self.edv is None or self.edv == self['edv']``

returns an AssertionError.  If this occurs because the VDIF EDV is not yet
supported by Baseband, support can be added by implementing a custom header
class.  If the EDV is supported, but the header deviates from the format
found in the `VLBI.org EDV registry <http://www.vlbi.org/vdif/>`_, the 
best solution is to create a custom header class, then override the
subclass selector in :class:`~baseband.vdif.header.VDIFHeader`.  Tutorials
for doing either can be found :ref:`here <new_edv>`.

EOFError encountered in ``_get_frame_rate`` when reading
--------------------------------------------------------

When the number of frames per second is not input by the user and cannot be
deduced from header information (if EDV = 1, 3 or 4, the frame rate can be
derived from the sampling rate found in the header), Baseband tries to
determine the frame rate using the private method ``_get_frame_rate`` in
`~baseband.vdif.base.VDIFStreamReader`.  This function raises
`EOFError` if the file contains less than one second of data, or is corrupt.
In either case the file can be opened still by explicitly passing in the frame
rate to :func:`~baseband.vdif.open` via the `frames_per_second` argument.

.. _vdif_api:

Reference/API
=============

.. automodapi:: baseband.vdif
.. automodapi:: baseband.vdif.header
   :include-all-objects:
.. automodapi:: baseband.vdif.payload
.. automodapi:: baseband.vdif.frame
.. automodapi:: baseband.vdif.base
