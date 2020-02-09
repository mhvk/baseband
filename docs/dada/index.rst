.. _dada:

****
DADA
****

Distributed Acquisition and Data Analysis (DADA) format data files contain a
single :term:`data frame` consisting of an ASCII :term:`header` of typically
4096 bytes followed by a :term:`payload`.  DADA is defined by its
`software specification
<http://psrdada.sourceforge.net/manuals/Specification.pdf>`_ and
actual usage; files are described by an :ref:`ASCII header <dada_header>`.

.. _dada_usage:

Usage
=====

This section covers reading and writing DADA files with Baseband; general usage
is covered in the :ref:`Using Baseband <using_baseband>` section. For
situations in which one is unsure of a file's format, Baseband features the
general `baseband.open` and `baseband.file_info` functions, which are also
discussed in :ref:`Using Baseband <using_baseband>`.  The examples below use
the sample file ``baseband/data/sample.dada``, and the the `astropy.units` and
`baseband.dada` modules::

    >>> from baseband import dada
    >>> import astropy.units as u
    >>> from baseband.data import SAMPLE_DADA

Single files can be opened with `~baseband.dada.open` in binary mode. DADA
files typically consist of just a single header and payload, and can be
read into a single `~baseband.dada.DADAFrame`.

::

    >>> fb = dada.open(SAMPLE_DADA, 'rb')
    >>> frame = fb.read_frame()
    >>> frame.shape
    (16000, 2, 1)
    >>> frame[:3].squeeze()
    array([[ -38.-38.j,  -38.-38.j],
           [ -38.-38.j,  -40. +0.j],
           [-105.+60.j,   85.-15.j]], dtype=complex64)
    >>> fb.close()

Since the files can be quite large, the payload is mapped (with
`numpy.memmap`), so that if one accesses part of the data, only the
corresponding parts of the encoded payload are loaded into memory (since the
sample file is encoded using 8 bits, the above example thus loads 12 bytes into
memory).

Opening in stream mode wraps the low-level routines such that reading and
writing is in units of samples, and provides access to header information::

    >>> fh = dada.open(SAMPLE_DADA, 'rs')
    >>> fh
    <DADAStreamReader name=... offset=0
        sample_rate=16.0 MHz, samples_per_frame=16000,
        sample_shape=SampleShape(npol=2), bps=8,
        start_time=2013-07-02T01:39:20.000>
    >>> d = fh.read(10000)
    >>> d.shape
    (10000, 2)
    >>> d[:3]
    array([[ -38.-38.j,  -38.-38.j],
           [ -38.-38.j,  -40. +0.j],
           [-105.+60.j,   85.-15.j]], dtype=complex64)
    >>> fh.close()

To set up a file for writing as a stream is possible as well::

    >>> from astropy.time import Time
    >>> fw = dada.open('{utc_start}.{obs_offset:016d}.000000.dada', 'ws',
    ...                sample_rate=16*u.MHz, samples_per_frame=5000,
    ...                npol=2, nchan=1, bps=8, complex_data=True,
    ...                time=Time('2013-07-02T01:39:20.000'))
    >>> fw.write(d)
    >>> fw.close()
    >>> import os
    >>> [f for f in sorted(os.listdir('.')) if f.startswith('2013')]
    ['2013-07-02-01:39:20.0000000000000000.000000.dada',
     '2013-07-02-01:39:20.0000000000020000.000000.dada']
    >>> fr = dada.open('2013-07-02-01:39:20.{obs_offset:016d}.000000.dada', 'rs')
    >>> d2 = fr.read()
    >>> (d == d2).all()
    True
    >>> fr.close()

Here, we have used an even smaller size of the payload, to show how one can
define multiple files.  DADA data are typically stored in sequences of files.
If one passes a time-ordered list or tuple of filenames to
`~baseband.dada.open`, it uses |sequentialfile.open| to access the sequence.
If, as above, one passes a template string, `~baseband.dada.open` uses
`~baseband.dada.base.DADAFileNameSequencer` to create and use a filename
sequencer.  (See API links for further details.)

.. |sequentialfile.open| replace:: `sequentialfile.open <baseband.helpers.sequentialfile.open>`

Further details
===============

.. toctree::
   :maxdepth: 1

   header

.. _dada_api:

Reference/API
=============

.. automodapi:: baseband.dada
.. automodapi:: baseband.dada.header
.. automodapi:: baseband.dada.payload
.. automodapi:: baseband.dada.frame
.. automodapi:: baseband.dada.base
