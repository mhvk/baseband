.. _glossary:

.. include:: glossary_substitutions.rst

********
Glossary
********

.. glossary::

   channel
      A single component of the :term:`complete sample`, or a :term:`stream`
      thereof.  They typically represent one frequency sub-band, the output
      from a single antenna, or (for channelized data) one spectral or Fourier
      channel, ie. one part of a Fourier spectrum.

   complete sample
      Set of all component samples - ie. from all threads, polarizations,
      channels, etc. - for one point in time.  Its dimensions are given by the
      :term:`sample shape`.

   component
      One individual :term:`thread` and :term:`channel`, or one polarization
      and channel, etc.  Component samples each occupy one element in decoded
      data arrays.  A component sample is composed of one :term:`elementary
      sample` if it is real, and two if it is complex.

   data frame
      A block of time-sampled data, or :term:`payload`, accompanied by a
      :term:`header`. "Frame" for short.

   data frameset
      In the VDIF format, the set of all |data frames| representing the same
      segment of time.  Each data frame consists of sets of |channels| from
      different |threads|.

   elementary sample
      The smallest subdivision of a complete sample, i.e. the real / imaginary
      part of one :term:`component` of a :term:`complete sample`.

   header
      Metadata accompanying a :term:`data frame`.

   payload
      The data within a :term:`data frame`.

   sample
      Data from one point in time.  |Complete samples| contain samples from all
      |components|, while |elementary samples| are one part of one component.

   sample rate
      Rate of complete samples.

   sample shape
      The lengths of the dimensions of the complete sample.

   squeezing
      The removal of any dimensions of length unity from decoded data.

   stream
      Timeseries of |samples|; may refer to all of, or a subsection of, the
      dataset.

   subset
      A subset of a complete sample, in particular one defined by the user for
      selective decoding.

   thread
      A collection of |channels| from the :term:`complete sample`, or a
      :term:`stream` thereof.  For VDIF, each thread is carried by a separate 
      (set of) :term:`data frame(s)<data frame>`.
