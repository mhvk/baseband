.. _new_format:

***********************
Supporting a New Format
***********************

To support a new format, it may be easiest to start with an existing format
that is as similar as possible to the new format, basing classes on those
provided in `~baseband.base`, in particular
:class:`~baseband.base.base.VLBIStreamReaderBase`.

To connect a new format to the baseband eco-system, it should be in its own
module and there should be an ``open`` function -- and ideally also an
``info`` function that checks whether the file is of the right format and
collects some basic information.  For these, the basebands formats themselves
construct callable instances with :class:`~baseband.base.base.FileOpener`
and :class:`~baseband.base.base.FileInfo`.

If one has constructed a module, it can be made available in baseband by
defining an entry point for ``baseband.io`` in ``setup.cfg``, e.g.::

  [options.entry_points]
  baseband.io =
      hdf5 = scintillometry.io.hdf5

With this, if a user has the package installed, :func:`baseband.open` and
:func:`baseband.file_info` will automatically recognize the format.

Of course, if the format is useful for others, it will definitely be
considered for inclusion in baseband proper!  More generally, feel free to ask
for help by raising an issue on the github page.

Example: supporting Arecibo Signal Processing format
====================================================

The ASP format does not seem to have a formal definition, but as it is
supported by dspsr, we can work from that code.  A given observation
consists of multiple files that each contain multiple blocks, with
each file having a largish file header and each block a small block
header.  This makes it a bit of a cross between DADA and GUPPI.  Both
headers can be mapped to C structs. In particular, the
`file header struct <https://github.com/UCBerkeleySETI/bl-dspsr/blob/1d3449c9511cebaaf32914ccdb9abcadc45ae0c1/Kernel/Formats/asp/asp_params.h#L11-L40>`_
is given by::

  struct asp_params {
    int32_t n_ds;
    int32_t n_chan;
    double ch_bw;
    double rf;
    int32_t band_dir;
    char psr_name[12];
    double dm;
    int32_t fft_len;
    int32_t overlap;
    int32_t n_bins;
    float t_dump;
    int32_t n_dump;
    int64_t n_samp_dump;
    int32_t imjd;   /* Note:  These are NOT precise obs start times, */
    double fmjd; /*        just a rough estimate to check that    */
                 /*        the polycos are valid                  */
    int32_t cal_scan;

    char scan[256];
    char observer[256];
    char proj_id[256];
    char comment[1024];
    char telescope[2];
    char front_end[256];
    char pol_mode[12];
    double ra;
    double dec;
    float epoch;
  }

And the `block header struct
<https://github.com/UCBerkeleySETI/bl-dspsr/blob/1d3449c9511cebaaf32914ccdb9abcadc45ae0c1/Kernel/Formats/asp/data2rcv.h#L11-L18>`_
is given by::

  struct data2rcv {
    int32_t totalsize;
    int32_t NPtsSend;
    double iMJD;
    double fMJD;
    int64_t ipts1,ipts2; /* Actual position of the start and end in the data time serie */
    int32_t  FreqChanNo;
  }

This is different from existing formats: the VLBI formats all encode
specific bits, while the DADA, GUPPI, and GSB formats are text-based.
Instead, this may be most easily captured by a structured
`numpy.dtype`, which are meant to match C structs.

In principle, like :class:`~baseband.dada.DADAHeader` is based on a
`dict`, we could base ourselves on a :class:`numpy.ndarray` or
:class:`numpy.void` with such a dtype, but then we inherit all their
methods, so instead it may be better to still think in terms of a
:class:`~baseband.base.header.ParsedHeader`, but override the getting
and setting of keywords using the built-in numpy methods.  Hence, we
first define a :class:`~baseband.asp.header.DTypeHeaderBase` and then use
this for :class:`~baseband.asp.header.ASPFileHeader` and
:class:`~baseband.asp.header.ASPBlockHeader`.
