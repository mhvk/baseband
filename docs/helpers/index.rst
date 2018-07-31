.. _helpers:

.. include:: ../tutorials/glossary_substitutions.rst

****************
Baseband Helpers
****************

Helpers assist with reading and writing all file formats.  Currently,
they only include the :mod:`~baseband.helpers.sequentialfile` module
for reading a sequence of files as a single one.

.. _sequential_file:

Sequential File
===============

The `~baseband.helpers.sequentialfile` module is for reading from and writing
to a sequence of files as if they were a single, contiguous one.  Like with
file formats, there is a master `sequentialfile.open
<baseband.helpers.sequentialfile.open>` function to open sequences either
for reading or writing.  It returns sequential file objects that have ``read``,
``write``, ``seek``, ``tell``, and ``close`` methods that work identically to
their single file object counterparts.  They additionally have ``memmap``
methods to read or write to files through `numpy.memmap`.

It is usually unnecessary to directly access `~baseband.helpers.sequentialfile`,
since it is used by `baseband.open` and all format openers (except GSB)
whenever a sequence of files is passed - see the :ref:`Using Baseband
documentation <using_baseband_multifile>` for details. For finer control of
file opening, however, one may manually create a
`~baseband.helpers.sequentialfile` object, then pass it to an opener.

To illustrate, we rewrite the multi-file example from :ref:`Using Baseband
<using_baseband_multifile>`.  We first load the required data::

    >>> from baseband import vdif
    >>> from baseband.data import SAMPLE_VDIF
    >>> import numpy as np
    >>> fh = vdif.open(SAMPLE_VDIF, 'rs')
    >>> d = fh.read()

We now create a sequence of filenames and calculate the byte size per file,
then pass these to `~baseband.helpers.sequentialfile.open`::

    >>> from baseband.helpers import sequentialfile as sf
    >>> filenames = ["seqvdif_{0}".format(i) for i in range(2)]
    >>> file_size = fh.fh_raw.seek(0, 2) // 2
    >>> fwr = sf.open(filenames, mode='w+b', file_size=file_size)

The first argument passed to |open| must be a **time-ordered sequence** of
filenames in a list, tuple, or other container that returns ``IndexError`` when
the index is out of bounds.  The read mode is 'w+b' (a requirement of all
format openers just in case they use `numpy.memmap`), and ``file_size``
determines the largest size a file may reach before the next one in the
sequence is opened for writing.  We set ``file_size`` such that each file holds
exactly one frameset.

To write the data, we pass ``fwr`` to `vdif.open <baseband.vdif.open>`::

    >>> fw = vdif.open(fwr, 'ws', header0=fh.header0,
    ...                sample_rate=fh.sample_rate,
    ...                nthread=fh.sample_shape.nthread)
    >>> fw.write(d)
    >>> fw.close()    # This implicitly closes fwr.

To read the sequence and confirm their contents are identical to the sample
file's, we may again use |open|::

    >>> frr = sf.open(filenames, mode='rb')
    >>> fr = vdif.open(frr, 'rs', sample_rate=fh.sample_rate)
    >>> fr.header0.time == fh.header0.time
    True
    >>> np.all(fr.read() == d)
    True
    >>> fr.close()
    >>> fh.close()  # Close sample file.

.. |open| replace:: `~baseband.helpers.sequentialfile.open`

Reference/API
=============

.. automodapi:: baseband.helpers
.. automodapi:: baseband.helpers.sequentialfile
