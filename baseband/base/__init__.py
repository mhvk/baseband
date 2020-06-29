# Licensed under the GPLv3 - see LICENSE
"""Base implementations shared between all formats.

Files are considered as composed of multiple frames, each of which have a
header and payload, which are encoded in various ways. Base classes
implementing the decoding and encoding and exposing a standardized
interface are found in the corresponding `~baseband.base.header`,
`~baseband.base.payload` and `~baseband.base.frame` modules, with the
separate `~baseband.base.encoding` module providing implementations for
common encoding formats.

The `~baseband.base.base` module defines base methods for file and stream
readers and writers that read or write the frames, including possibly
dealing with corrupted data, using the `~baseband.base.offsets` module to
keep track of missing pieces.  Each file and stream reader has an ``info``
property, defined in `~baseband.base.file_info`, that provides standardized
information.

Finally, `~baseband.base.utils` contains some general utility routines such
as for BCD encoding and decoding, and cyclic redundancy checks.

"""
