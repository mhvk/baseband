from ..base.base import (  # noqa
    HeaderNotFoundError,
    FileBase as _FileBase,
    VLBIFileReaderBase,
    StreamBase as _StreamBase,
    VLBIStreamReaderBase,
    StreamWriterBase as _StreamWriterBase,
    FileOpener as _FileOpener)


class VLBIFileBase(_FileBase):
    pass


class VLBIStreamBase(_StreamBase):
    pass


class VLBIStreamWriterBase(_StreamWriterBase):
    pass


def make_opener(fmt, classes, doc='', append_doc=True):
    opener = _FileOpener.create(classes, doc)
    if not append_doc:
        opener.__doc__ = doc
    return opener
