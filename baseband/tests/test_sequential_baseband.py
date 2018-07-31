# Licensed under the GPLv3 - see LICENSE
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import numpy as np
import astropy.units as u
from astropy.time import Time
from .. import vdif
from ..helpers import sequentialfile
from ..helpers.tests.test_sequentialfile import Sequencer


def test_sequentialfile_vdif_stream(tmpdir):
    vdif_sequencer = Sequencer(str(tmpdir.join('{:07d}.vdif')))
    # Try writing a very simple file, using edv=0.
    data = np.ones((16, 16, 2, 2))
    for i, dat in enumerate(data):
        dat[i, 0, 0] = -1.
        dat[i, 1, 1] = -1.
    data.shape = -1, 2, 2
    # construct first header
    header = vdif.VDIFHeader.fromvalues(
        edv=0, time=Time('2010-01-01'), nchan=2, bps=2,
        complex_data=False, frame_nr=0, thread_id=0, samples_per_frame=16,
        station='me')
    with sequentialfile.open(vdif_sequencer, 'w+b',
                             file_size=4*header.frame_nbytes) as sfh, \
            vdif.open(sfh, 'ws', header0=header, nthread=2,
                      sample_rate=256*u.Hz) as fw:
        fw.write(data)
    # check that this wrote 8 frames
    files = [vdif_sequencer[i] for i in range(8)]
    for file_ in files:
        assert os.path.isfile(file_)
    assert not os.path.isfile(vdif_sequencer[8])

    with sequentialfile.open(vdif_sequencer, 'rb') as sfh, vdif.open(
            sfh, 'rs', sample_rate=256*u.Hz) as fr:
        record1 = fr.read(21)
        assert np.all(record1 == data[:21])
        fr.seek(7*16)
        record2 = fr.read(61)
        assert np.all(record2 == data[7*16:7*16+61])
        assert fr.tell() == 7*16+61
    # Might as well check file list too.
    with sequentialfile.open(files, 'rb') as sfh, vdif.open(
            sfh, 'rs', sample_rate=256*u.Hz) as fr:
        record1 = fr.read()
        assert np.all(record1 == data)
