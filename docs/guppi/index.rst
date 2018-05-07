*****
GUPPI
*****

The GUPPI format is the output of the `Green Bank Ultimate Pulsar Processing
Instrument <https://safe.nrao.edu/wiki/bin/view/CICADA/NGNPP>`_ and any clones
operating at other telescopes, such as `PUPPI at the Arecibo Observatory
<http://www.naic.edu/puppi-observing/>`_.  Baseband specifically supports GUPPI
data **taken in baseband mode**, and is based off of `DSPSR's
<https://github.com/demorest/dspsr>`_.  While general format specifications can
be found at the `SERA Project
<http://seraproject.org/mw/index.php?title=GBT_FIle_Formats>`_ and on `Paul
Demorest's site <https://www.cv.nrao.edu/~pdemores/GUPPI_Raw_Data_Format>`_,
some of the header information could be invalid or not applicable.

Baseband currently only supports 8-bit |elementary samples|.

.. _guppi_file_structure:

File Structure
==============

Each GUPPI file contains multiple (typically 128) |frames|, with each frame
consisting of an ASCII :term:`header` composed of 80-character entries,
followed by a binary :term:`payload` (or "block").  The header's length is
variable, but always ends with "END" followed by 77 spaces.

The payload stores each channel's :term:`stream` in a contiguous block of
samples, rather than grouping the components of a :term:`complete sample`
together.  For each channel, polarization samples from the same point in time,
are stored adjacent to one another.  At the end of each channel's stream is a
section of **overlap samples** identical to the first samples in the next
payload.  Baseband retains these redundant samples when reading individual
GUPPI frames, but removes them when reading files as a stream.

.. _guppi_usage:

Usage
=====
