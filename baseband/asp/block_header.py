from ..vlbi_base.header import (four_word_struct, eight_word_struct,
                                HeaderParser, VLBIHeaderBase)

class ASPBlockHeader(VDIFHeader):
	# _edv = ?
	# is this necessary? this is not a vdif edv

	_edv=99 #?

	_header_parser = HeaderParser(
		(('totalsize', (0, 0, 32)),
		 ('NPtsSend', (1, 0, 32)),
		 ('iMJD', (2, 0, 64)),
		 ('fMJD', (4, 0, 64)),
		 ('ipts1', (6, 0, 64)),
		 ('ipts2', (8, 0, 64)),
		 ('FreqChanNo', (10, 0, 32)),
		))