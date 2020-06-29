from ..base.base import (  # noqa
    HeaderNotFoundError,
    FileBase as _FileBase,
    VLBIFileReaderBase,
    StreamBase as _StreamBase,
    VLBIStreamReaderBase as _VLBIStreamReaderBase,
    StreamWriterBase as _StreamWriterBase,
    FileOpener as _FileOpener)


class VLBIFileBase(_FileBase):
    pass


class _UnslicedShape:
    def __init__(self, *args, **kwargs):
        if 'unsliced_shape' in kwargs:
            kwargs['sample_shape'] = kwargs.pop('unsliced_shape', None)
        super().__init__(*args, **kwargs)


class VLBIStreamBase(_UnslicedShape, _StreamBase):
    pass


class VLBIStreamReaderBase(_UnslicedShape, _VLBIStreamReaderBase):
    pass


class VLBIStreamWriterBase(_UnslicedShape, _StreamWriterBase):
    pass


def make_opener(fmt, classes, doc='', append_doc=True):
    opener = _FileOpener.create(classes, doc)
    if not append_doc:
        opener.__doc__ = doc
    return opener
