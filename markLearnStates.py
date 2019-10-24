import numpy as np
import random as rd

import config
import addDendrites

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
        # [ToDo: check if shape of activeCols is (,n) => use activeCols[k] or (1,n) => use activeCols[0][k]
        j = activeCols[k]
        # Find the row indices (row_i) of active cells in column j
        i = row_i[col_i == j] # i can have more than one value, it can be a vector
        [cellChosen, newSynapsesToDendrite, updateFlag] = getBestMachingCell(j, i)

        # If the column is shared between two time instant, use the locations chosen earlier.
        if ((updateFlag == True) and (newSynapsesToDendrite < 0)):
            xJ = np.nonzero(cellColPrev == j)
            if (np.size(xJ) > 0):
                cellChosen = cellIDPrevious[xJ[0][0]]
        config.SM['cellLearn'][np.unravel_index(cellChosen, (config.SM['M'], config.SM['N']), 'F')] = True
        if updateFlag:
            dCells[nDCells] = cellChosen
            expandDendrites[nDCells] = newSynapsesToDendrite
            nDCells = nDCells + 1

    addDendrites.addDendrites(dCells, expandDendrites, nDCells)






def getBestMachingCell(j, i):
    """
    i could be a vector - is the list of active cells (could be bursting) in the column, j.
    getBestMatchingCell - For the given column, return the cell with the best matching segment (as defined below).
    If no cell has a matching segment, then return a cell with the fewest number of segments using a random tie breaker

    Best matching segment - For the given column j cells i, find the segment with the largest number of ACTIVE synapses.
    This routine is aggressive in finding the best match. The permanence value of synapses is ALLOWED to be below connectedPERM.
    The number of active synapses is allowed to be below activationThreshold, but must be above minThreshold.
    The routine returns the segment index.
    If no segments are found, then an index of -1 is returned.

    we can have  one active cell -- choose it to a potential synapse for next cycle
    can more than one active cell -- choose the "best" active cell -- the one with maximum positive dentritic connection.
    can have bursting column with or without any dendrities, e.g. at the start of a new
    sequence, randomly choose one -- this will also be an anchor for a new dendrite.

    :param j:   Column number.
    :param i:   Row indices in column j.
    :return:
                chosenCell: cell ID chosen from jth column in (config.SM['M'] x config.SM['N']) shaped array
                addNewSynapsesToDendrite: dendrite id to be added or -1
                updateFlag: True or False
    """
    cellIndex = np.ravel_multi_index((i, j*np.ones(np.shape(i), dtype=int)), (config.SM['M'], config.SM['N']), order='F')
    dendrites_temp = np.isin(config.SM['dendriteToCell'], cellIndex)
    [den_r, den_c] = np.nonzero(dendrites_temp)
    dendrites = np.ravel_multi_index((den_r, den_c), np.shape(config.SM['dendriteToCell']), order='F')
    lcChosen = False
    addNewSynapsesToDendrite = -1
    updateFlag = False

    if  (np.shape(dendrites)!=0):
        id = np.argmax(config.SM['dendritePositive'][dendrites])
        val = config.SM['dendritePositive'][dendrites[id]][0]
        if (val > config.SM['minPositiveThreshold']):
            chosenCell = config.SM['dendriteToCell'][dendrites[id]][0]
            lcChosen = True
            if (val < config.SM['Theta']):
                addNewSynapsesToDendrite = dendrites[id]
                updateFlag = True

    # Add new dendrite if no dendrites of the active cells are above minimum threshold.
    if (lcChosen == False):
        # Randomly choose location to add a dendrite.
        ndpc_ind = np.unravel_index(cellIndex, (config.SM['M'], config.SM['N']), order='F')
        args = np.argsort(config.SM['numDendritesPerCell'][ndpc_ind])
        sorted_cellIndexs = np.unravel_index(cellIndex[args], (config.SM['M'], config.SM['N']), order='F')
        val = config.SM['numDendritesPerCell'][sorted_cellIndexs]
        tie = (val == val[0])
        rid = rd.randint(0, np.sum(tie))
        chosenCell = cellIndex[args[rid]]
        updateFlag = True
    return [chosenCell, addNewSynapsesToDendrite, updateFlag]


