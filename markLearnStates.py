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

    # Structure of cells in colums:
    # Each column has "config.SM['M']" cells. Let us assume "config.SM['M'] = 32"
    # Cells in columns are numbers in column major fashion starting from zero:
    # i.e second cell in first column is numbered 1. last cell in first column is numbered 31.
    # First cell in second column is numbered 32..

    config.SM['cellLearn'][:] = 0
    activeCols = np.nonzero(config.SM['input'])

    """ Mark the correctly predicted active cells with dendrites that are also
        predictive based on on learning states.  """
    xL = np.nonzero(config.SM['dendriteLearn'])[0]  # active learning dendrites
    uL = np.unique(config.SM['dendriteToCell'][xL]) # Cell to which active learning dendrites are connected

    # If predictedActive is non-empty array
    if np.prod(np.shape(config.SM['predictedActive'])) != 0:
        # Cells that are connected to Active learning dendrites and are predicted active.
        # lc_cols: contains the cell numbers in column major
        lc_cols = uL[config.SM['predictedActive'][np.unravel_index(uL, np.shape(config.SM['predictedActive']), 'F')] > 0]
        [R, C] = np.unravel_index(lc_cols, (config.SM['M'], config.SM['N']), 'F')
        # select only one active cell per column
        [C, IA] = np.unique(C, return_index = True)
        R = R[IA]
        config.SM['cellLearn'][tuple(R), tuple(C)] = True
    else:
        lc_cols = np.empty([])


    # lc_cols: represents the cell numbers, extract only column numbers of these cells
    # following line assumes that np.shape(config.SM['predictedActive']) is equal to (config.SM['M'], config.SM['N'])
    [r_num, c_num] = np.unravel_index(lc_cols, (config.SM['M'], config.SM['N']), 'F')
    unique_lc_cols = np.unique(c_num)
    # Find the active columns without a learnCell state set -- activeCols
    activeCols = np.setdiff1d(activeCols, unique_lc_cols)

    # Iterate through the remaining columns selecting a single learnState cell in each
    n = np.size(activeCols)
    [row_i, col_i] = np.nonzero(config.SM['cellActive'])
    [cellRowPrev, cellColPrev] = np.nonzero(config.SM['cellLearnPrevious'])
    cellIDPrevious = np.ravel_multi_index((cellRowPrev, cellColPrev), np.shape(config.SM['cellLearnPrevious']), order='F')
    dCells = np.zeros((config.SM['N'], 1))
    nDCells = 0
    expandDendrites = np.zeros(config.SM['N'], 1)

    for k in range(0, n):
        # Iterate through columns looking for cell to set learnState
        j = activeCols[k]




