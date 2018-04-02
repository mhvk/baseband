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

As an example of how to use |open|, we write the data from the sample VDIF
file ``baseband/data/sample.vdif`` into a sequence of two files - as the sample
file has two |framesets| - and then read the files back in.  We first load the
required data::

    >>> from baseband import vdif
    >>> from baseband.data import SAMPLE_VDIF
    >>> import numpy as np
    >>> fh = vdif.open(SAMPLE_VDIF, 'rs')
    >>> d = fh.read()

We now open a sequential file object for writing::

    >>> from baseband.helpers import sequentialfile as sf
    >>> filenames = ["seqvdif_{0}".format(i) for i in range(2)]
    >>> file_size = fh.fh_raw.seek(0, 2) // 2
    >>> fwr = sf.open(filenames, mode='wb', file_size=file_size)

The first argument passed to |open| must be a **time-ordered sequence** of
filenames in a list, tuple, or other subscriptable object that returns
``IndexError`` when the index is out of bounds.  The read mode is 'wb',
though note that writing using `numpy.memmap` (eg. required for the DADA stream
writer) is only possible if ``mode='w+b'``.  ``file_size`` determines the
largest size a file may reach before the next one in the sequence is opened
for writing.  We set ``file_size`` such that each file holds exactly one
frameset.

.. note::

    Setting ``file_size`` to a larger value than above will lead to the
    two files having different sizes.  By default, ``file_size=None``, meaning
    it can be arbitrarily large, in which case only one file will be created.

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

We can also open the second file on its own and confirm it contains the second
frameset of the sample file::

    >>> fsf = vdif.open(filenames[1], mode='rs', sample_rate=fh.sample_rate)
    >>> fh.seek(fh.size // 2)    # Seek to start of second frameset.
    20000
    >>> fsf.header0.time == fh.time
    True
    >>> np.all(fsf.read() == fh.read())
    True
    >>> fsf.close()
    >>> fh.close()  # Close sample file.

While `~baseband.helpers.sequentialfile` can be used for any format, since
file sequences are common for DADA, it is implicitly used if a list of
files or filename template is passed to `dada.open <baseband.dada.open>`. 
See the DADA :ref:`Usage <dada_usage>` section for details.

.. |open| replace:: `~baseband.helpers.sequentialfile.open`

Reference/API
=============

.. automodapi:: baseband.helpers
.. automodapi:: baseband.helpers.sequentialfile
