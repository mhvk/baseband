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
