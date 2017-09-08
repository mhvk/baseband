.. _frame_io:

*******************************
Reading and Writing Data Frames
*******************************

Data Frames and Frame Sets
==========================

Radio baseband files and transmissions are composed of sequences of data
frames, each with a header containing metadata followed by (or concurrent with)
a data array, or "payload".  Appropriately, for each data format
it supports, Baseband defines a class for representing the header and one for
representing payload, and also defines a frame class that encloses both. While
frames and their constituent headers and payloads are abstracted away when using
each format's stream reader and writer, the size and shape of a frame must at
least be defined for writing (see the :ref:`Getting Started writing tutorial
<getting_started_writing>`).  Finding and reading individual frames may also be
necessary when reading files with errors.

This tutorial describes how to find, access and write individual data frames,
using VDIF as an example.  It assumes Numpy has been imported::

    >>> import numpy as np

Binary File Pointer
===================

All of Baseband's readers and writers use Python's native `io.BufferedReader`
class to read files.  When ``open`` for a file format is called, for example, it
first calls `io.open`, then passes the returned `~!io.BufferedReader` object to
the format's file or stream reader.  This object can be accessed through the
``fh_raw`` attribute of the reader.  For example, for VDIF::

    >>> import baseband.vdif as vdif
    >>> from baseband.data import SAMPLE_VDIF
    >>> fh = vdif.open(SAMPLE_VDIF, 'rs')
    >>> fh.fh_raw
    <_io.BufferedReader name=...>

The `~!io.BufferedReader` has a binary file pointer, which is accessible
with the same syntax as the sample file pointer, except it only accepts, and
reports back in, units of bytes::

    >>> fh.fh_raw.tell()  # Non-zero because reader already read a frame set
    40256
    >>> fh.fh_raw.seek(0, 2)
    80512
    >>> fh.close()

Stream readers include the sample file pointer discussed in :ref:`Getting Started
<getting_started_reading>`, which allows for easy seeking of data samples in
time.  The binary pointer, though, is what is actually used to read files in
the backend, and only it can be used to read frames.

.. note::

    The binary and sample file pointers do **not** automatically track one
    another.  Mixing them is not advised.  For example::

        >>> fh = vdif.open(SAMPLE_VDIF, 'rs')
        >>> fh.fh_raw.seek(10)  # Move binary pointer to 10th sample
        10
        >>> fh.tell()           # Sample pointer still at first sample!
        0
        >>> fh.close()

Reading Frames
==============

To read a frame, we use the binary pointer to seek to the start of it, then call
the ``fromfile`` method of the frame class.  The size in bytes of one frame
is included in the header.  If we use the stream reader (as above), the first
header of the file is accessible through the ``header0`` attribute.  However, if

For example, if we wished to extract the
11th frame header from the sample VDIF, we would::

    >>> fh = vdif.open(SAMPLE_VDIF, 'rs')
    >>> fh.fh_raw.seek(11 * fh.header0.framesize)
    55352

file to check it for errors, we can seek to it, then 

Finally::

    >>> fh.close()