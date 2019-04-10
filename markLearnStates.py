import numpy as np

import config

def markLearnStates():
    """"
    Update the learn states of the cells (one per ACTIVE columns). This is to be run after the active states
    have been updated (compute_active_states). For those ACTIVE COLUMNS, this code further selects ONE cell
    per column as the learning cell (learnState). The logic is as follows. If an active cell has a segment that
    became active from cells chosen with learnState on, this cell is selected as the learning cell, i.e. learnState is
    set to 1.

    For bursting columns, the best matching cell is chosen as the learning cell and a new segment is added to that
    cell. Note that it is possible that there is no best matching cell; in this case getBestMatchingCell chooses
    a cell with the fewest number of segments, using a random tiebreaker

    getBestMatchingCell - For the given column, return the cell with the best matching segment (as defined below).
    If no cell has a matching segment, then return a cell with the fewest number of segments using a
    random tiebreaker.

    Best matching segment - For the given column c cell i, find the segment with the largest number of ACTIVE
    synapses. This routine is aggressive in finding the best match. The permanence value of
    synapses is ALLOWED to be below connectedPerm. The number of active synapses is allowed to
    be below activationThreshold, but must be above minThreshold. The routine returns the
    segment index. If no segments are found, then an index of -1 is returned.
    """
    config.SM['cellLearn'][:] = 0
    activeCols =
