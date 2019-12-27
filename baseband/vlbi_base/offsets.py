# Licensed under the GPLv3 - see LICENSE
import bisect
import operator


class RawOffsets:
    """File offset tracker.

    Keeps track of offsets from expected file position as a
    function of frame number, by keeping a joint list, of the
    first frame number beyond which a certain offset will hold.

    The offset for a given frame number is retrieved via ``__getitem__``,
    while new offsets are added via ``__setitem__`` (keeping the
    list of frame numbers minimal for identical offsets).

    Parameters
    ----------
    frame_nr : list
        Frame number for which one has offsets.
    offset : list
        Corresponding offsets.
    frame_nbytes : int
        Possible frame size to include in all returned offsets, i.e.,
        output will be ``offset + index * frame_nbytes``.  Default: 0.

    Examples
    --------
    The usage is best seen through an example::

      >>> from baseband.vlbi_base.offsets import RawOffsets
      >>> offsets = RawOffsets([6], [5])
      >>> offsets[3]  # Implicitly 0 before first entry
      0
      >>> offsets[10]  # Assumed the same as 6
      5
      >>> offsets[10] = 9  # But suppose we find 10 has 9.
      >>> offsets[10]  # Then it takes that
      9
      >>> offsets[9]  # But before is still 5.
      5
      >>> offsets[8] = 9  # But if we find 8 has 9 too.
      >>> offsets[9]  # Then 9 is the same.
      9
      >>> offsets  # And the list is kept minimal.
      RawOffsets(frame_nr=[6, 8], offset=[5, 9], frame_nbytes=0)
      >>> offsets[9] = 9  # This is a no-op.
      >>> offsets[10] = 10  # But this is not.
      >>> offsets
      RawOffsets(frame_nr=[6, 8, 10], offset=[5, 9, 10], frame_nbytes=0)

    Similarly, if a frame size is passed in::

      >>> offsets = RawOffsets([6, 8, 10], [5, 9, 10], frame_nbytes=1000)
      >>> offsets
      RawOffsets(frame_nr=[6, 8, 10], offset=[5, 9, 10], frame_nbytes=1000)
      >>> offsets[1]
      1000
      >>> offsets[8]
      8009
      >>> offsets[8] = 8005  # This removes the offset for 9.
      >>> offsets[8]
      8005
      >>> offsets
      RawOffsets(frame_nr=[6, 10], offset=[5, 10], frame_nbytes=1000)

    """

    def __init__(self, frame_nr=None, offset=None, frame_nbytes=0):
        if frame_nr is None:
            frame_nr = []
        if offset is None:
            offset = []
        if len(frame_nr) != len(offset):
            raise ValueError('must have equal number of frame numbers '
                             'and offsets.')
        self.frame_nr = frame_nr
        self.offset = offset
        self.frame_nbytes = operator.index(frame_nbytes)

    def __getitem__(self, frame_nr):
        # Keep default case of no offsets as fast as possible.
        base = frame_nr * self.frame_nbytes
        if not self.frame_nr:
            return base
        # Find first index for which frame_nr < value-at-index,
        # hence the offset at the previous index is the one we need.
        index = bisect.bisect_right(self.frame_nr, frame_nr)
        return base if index == 0 else base + self.offset[index - 1]

    def __setitem__(self, frame_nr, offset):
        # Get the offset from expected.
        offset -= frame_nr * self.frame_nbytes
        # Find first index for which frame_nr < value-at-index.
        # Hence, this is where we should be if we do not yet exist.
        index = bisect.bisect_right(self.frame_nr, frame_nr)
        # If the entry already exists (should not really happen),
        # and the new value is different, replace it.
        if index > 0 and self.frame_nr[index-1] == frame_nr:
            if self.offset[index-1] == offset:
                return

            # Best to *remove* the entry, since the new value may
            # be consistent with one of the surrounding values,
            # in which case we can shorten our list.
            self.frame_nr.pop(index-1)
            self.offset.pop(index-1)
            index -= 1

        # If the offset at the next location is the same as ours,
        # then we can keep everything consistent by just moving the
        # frame number to our value.
        if index < len(self) and self.offset[index] == offset:
            self.frame_nr[index] = frame_nr
        elif index == 0 or self.offset[index-1] != offset:
            # Otherwise, if we add new information, insert ourserlves.
            self.frame_nr.insert(index, frame_nr)
            self.offset.insert(index, offset)

    def __len__(self):
        return len(self.frame_nr)

    def __repr__(self):
        return ('{0}(frame_nr={1}, offset={2}, frame_nbytes={3})'
                .format(type(self).__name__, self.frame_nr, self.offset,
                        self.frame_nbytes))
