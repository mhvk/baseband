# Licensed under the GPLv3 - see LICENSE
import pytest

from ..offsets import RawOffsets


class TestRawOffsets:
    @pytest.mark.parametrize('frame_nbytes', [0, 10, 100])
    def test_plain(self, frame_nbytes):
        ro = RawOffsets(frame_nbytes=frame_nbytes)
        assert ro[0] == 0
        assert ro[1] == frame_nbytes
        assert ro[10] == 10*frame_nbytes
        assert len(ro) == 0

    @pytest.mark.parametrize('frame_nr,offset', [
        ([10], [5]),
        ([5, 15], [-1, 1])])
    def test_with_list(self, frame_nr, offset):
        ro = RawOffsets(frame_nr=frame_nr, offset=offset)
        assert len(ro) == len(frame_nr)
        assert ro[11] == offset[0]

    @pytest.mark.parametrize('invalid_frame_nbytes', [1.5, (4,)])
    def test_invalid_frame_nbytes(self, invalid_frame_nbytes):
        with pytest.raises(TypeError):
            RawOffsets(frame_nbytes=invalid_frame_nbytes)

    @pytest.mark.parametrize('frame_nr,offset', [
        ([1], None),
        ([5, 15], [0])])
    def test_invalid_lists(self, frame_nr, offset):
        with pytest.raises(ValueError):
            RawOffsets(frame_nr=frame_nr, offset=offset)

    @pytest.mark.parametrize('frame_nbytes', [0, 10])
    def test_set_offsets(self, frame_nbytes):
        ro = RawOffsets(frame_nbytes=frame_nbytes)
        assert ro[10] == 10*frame_nbytes
        ro[10] = 1 + 10*frame_nbytes
        assert len(ro) == 1
        assert ro.frame_nr == [10]
        assert ro.offset == [1]
        assert ro[10] == 1 + 10*frame_nbytes
        assert ro[11] == 1 + 11*frame_nbytes
        assert ro[9] == 0 + 9*frame_nbytes
        assert ro[15] == 1 + 15*frame_nbytes
        ro[14] = 10 + 14*frame_nbytes
        assert len(ro) == 2
        assert ro.frame_nr == [10, 14]
        assert ro.offset == [1, 10]
        assert ro[15] == 10 + 15*frame_nbytes
        # If we add no new information, ignore
        ro[12] = 1 + 12*frame_nbytes
        assert len(ro) == 2
        assert ro.frame_nr == [10, 14]
        assert ro.offset == [1, 10]
        # Though if it precedes a frame with the same offset, we move.
        ro[8] = 1 + 8*frame_nbytes
        assert ro.frame_nr == [8, 14]
        assert ro.offset == [1, 10]
        assert ro[9] == 1 + 9*frame_nbytes
