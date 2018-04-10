*****
GUPPI
*****

The GUPPI format is the output of the `Green Bank Ultimate Pulsar Processing
Instrument <https://safe.nrao.edu/wiki/bin/view/CICADA/NGNPP>`_ and any clones
operating at other telescopes, such as `PUPPI at the Arecibo Observatory
<http://www.naic.edu/puppi-observing/>`_.  Like for DADA, each GUPPI
:term:`frame` consists of an ASCII :term:`header` composed of 80-character
entries, followed by a binary :term:`payload` (or "block").  The
header's length is variable, but always ends with "END" followed by 77 spaces. 
There are multiple frames per file.

Baseband's GUPPI module is based off of `DSPSR's <https://github.com/demorest/dspsr>`_.
It only reads GUPPI data taken in baseband mode.  For this mode, some of the
header information may be invalid or not applicable.

.. _guppi_usage:

Usage
=====
