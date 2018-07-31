from .header import ASPFileHeader, ASPHeader
from .payload import ASPPayload
from ..vlbi_base.frame import VLBIFrameBase

__all__ = ['ASPFrame']

class ASPFrame(VLBIFrameBase):

	_header_class = ASPHeader
	_payload_class = ASPPayload

	@classmethod
	def fromfile(cls, fh, valid=True, verify=True):
		header = cls._header_class.fromfile(fh, verify=verify)
		payload = cls._payload_class.fromfile(fh, header)
		return cls(header, payload, valid=valid, verify=verify)