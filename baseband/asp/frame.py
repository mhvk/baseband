from .header import ASPHeader
from .payload import ASPPayload
from ..base.frame import FrameBase


__all__ = ['ASPFrame', ]


class ASPFrame(FrameBase):

    _header_class = ASPHeader
    _payload_class = ASPPayload

    @classmethod
    def fromfile(cls, fh, file_header=None, memmap=None, valid=None,
                 verify=True, **kwargs):
        """Read a frame from a filehandle.

        Parameters
        ----------
        fh : filehandle
            Handle to read the frame from
        file_header : `~baseband.asp.header.ASPFileHeader`, optional
            Possible file header to attach to the block header and frame.
        memmap : bool, optional
            If `False`, read payload from file.  If `True`, map the payload
            in memory (see `~numpy.memmap`).  Only useful for large payloads.
            Default: as set by payload class.
        valid : bool, optional
            Whether the data are valid.  Default: inferred from header or
            payload read from file if possible, otherwise `True`.
        verify : bool, optional
            Whether to do basic verification of integrity.  Default: `True`.
        **kwargs
            Extra arguments that help to initialize the payload.
        """
        header = cls._header_class.fromfile(fh, file_header=file_header,
                                            verify=verify)
        payload = cls._payload_class.fromfile(fh, header=header,
                                              memmap=memmap, **kwargs)
        return cls(header, payload, valid=valid, verify=verify)
