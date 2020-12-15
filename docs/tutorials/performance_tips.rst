.. _performance_tips:

.. include:: ../tutorials/glossary_substitutions.rst

****************
Performance Tips
****************

Reading and decoding the data stored in baseband files can be somewhat slow,
especially if the analysis itself is simple.  So far, code development
has focussed more on correctness than on performance, but a few things
can help.

.. note:: If you have other tips on performance or contributions that
          help improve it, please raise an issue or make a pull request!

Minimize Verification
=====================

Once you know a file does not have missing frames or is otherwise
slighly corrupted, you can speed up reading by turning off
verification of headers, by passing in ``verify=False`` when opening
the stream reader.

Parallel Processing
===================

Like python file readers in general, baseband's stream readers cannot
be used in parallel threads.  They can, however, be sent from process
to process using `pickle <https://docs.python.org/3/library/pickle.html>`_
(or copied using `copy.deepcopy`).
