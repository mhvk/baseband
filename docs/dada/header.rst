.. _dada_header:

DADA Headers
************

The specification of "Distributed Acquisition and Data Analysis"
(DADA) headers is part of the `DADA software specification
<http://psrdada.sourceforge.net/manuals/Specification.pdf>`_.  In
particular, its appendix B.3 defines expected header keywords, which
we reproduce below.  We separate those for which the meaning has been
taken from comments in an `actual DADA header<baseband.data.SAMPLE_DADA>`
from Effelsberg, as well as additional keywords found in that header
that do not appear in the specification.

==================  ==============================================
Keyword             Description
==================  ==============================================
**Primary (from appendix B.3 [Default])**
------------------------------------------------------------------

HEADER              name of the header [DADA]
HDR_VERSION         version of the header [1.0]
HDR_SIZE            size of the header in bytes [4096]
INSTRUMENT          name of the instrument
PRIMARY             host name of the primary node on which the data were acquired
HOSTNAME            host name of the machine on which data were written
FILE_NAME           full path of the file to which data were written
FILE_SIZE           requested size of data files
FILE_NUMBER         number of data files written prior to this one
OBS_ID              the identifier for the observations
UTC_START           rising edge of the first sample (yyyy-mm-dd-hh:mm:ss)
MJD_START           the MJD of the first sample in the observation
OBS_OFFSET          the number of bytes from the start of the observation
OBS_OVERLAP         the amount by which neighbouring files overlap

**Secondary (description from Effelsberg sample file)**
------------------------------------------------------------------

TELESCOPE           name of the telescope
SOURCE              source name
FREQ                observation frequency
BW                  bandwidth in MHz (-ve lower sb)
NPOL                number of polarizations observed
NBIT                number of bits per sample
NDIM                dimension of samples (2=complex, 1=real)
TSAMP               sampling interval in microseconds
RA                  J2000 Right ascension of the source (hh:mm:ss.ss)
DEC                 J2000 Declination of the source (ddd:mm:ss.s)

**Other (found in Effelsberg sample file)**
------------------------------------------------------------------

PIC_VERSION         Version of the PIC FPGA Software [1.0]
RECEIVER            frontend receiver
SECONDARY           secondary host name
NCHAN               number of channels here
RESOLUTION     	    a parameter that is unclear
DSB                 (no description)
==================  ==============================================
