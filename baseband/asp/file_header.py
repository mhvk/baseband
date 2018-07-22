from ..vlbi_base.header import (four_word_struct, eight_word_struct,
                                HeaderParser, VLBIHeaderBase)

class ASPFileHeader(VDIFHeader):
	_edv=100 #?

	# uses 32-bit alignment
	_header_parser = HeaderParser(
		(('n_ds', (0, 0, 32)),
		 ('n_chan', (1, 0, 32)),
		 ('ch_bw', (2, 0, 64)),
		 ('rf', (4, 0, 64)),
		 ('band_dir', (6, 0, 32)),
		 ('psr_name', (7, 0, 24) ),
		 ('dm', (8, 0, 64)),
		 ('fft_len', (10, 0, 32)),
		 ('overlap', (11, 0, 32)),
		 ('n_bins', (12, 0, 32)),
		 ('t_dump', (13, 0, 32)),
		 ('n_dump', (14, 0 ,32)),
		 ('n_samp_dump', (15, 0, 64)),
		 ('imjd', (17, 0, 32)),
		 ('fmjd', (18, 0, 64)),

		 ('cal_scan', (20, 0, 32)),

		 ('scan', (21, 0, 256 * 8)),
		 ('observer', (85, 0, 256 * 8)),
		 ('proj_id', (149, 0, 256 * 8)),
		 ('comment', (213, 0, 1024 * 8)),
		 ('telescope', (469, 0, 2 * 8)),
		 ('front_end', (470, 0, 256 * 8)),
		 ('pol_mode', (534, 0, 12 * 8)),
		 ('ra', (537, 0, 64)),
		 ('dec', (539, 0, 64)),
		 ('epoch', ( 541, 0, 32))
		))