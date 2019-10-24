import numpy as np

import config

def markPredictiveStates():
    """
    The calculates the predictive state for each cell. A cell will turn on its predictive state output
    if one of its segments becomes active, i.e. if enough of its lateral inputs are currently active due
    to feed-forward input. In this case, the cell queues up the following changes:
    reinforcement of the currently active segment


    This function computes the indices of the cells in predicted state and stores it in the sparse array
    SM.cellPredicted. In the process of computing this, it stores the count of active synapses for the dendrites
    feeding from the currently  active cells. It keeps two kinds of counts ? count of active synapses in
    SM.dendriteActive and count of positive synapses in SM.dendritePositive.
    We also compute the list of dendrites that are active based on learnStates in SM.dendriteLearn

    :return:
    """
    # Initialize the active and positive synapse and dendrite to NULL sets
    # Note --- the underlying synapse/dendrite/cell pointer-based data structure is oriented in the following way
    # Cell body affecting (input) <-- synapse --> dendrites?--> cell body affected (output)
    config.SM['synapseActive'] = []  # synapses that have permanences above a threshold
    config.SM['synapsePositive'] = []  # synapses that have positive permanences
    config.SM['synapseLearn'] = []  # synapses that have positive permanences

    config.SM['dendriteActive'][:] = 0
    config.SM['dendritePositive'][:] = 0
    config.SM['dendriteLearn'][:] = 0

    # Mark the synapses that are on cells with active input and also those that could be potentially active
    # had their (positive) permanence been higher.

    # synapse is an index of the synapses along with corresponding cell body it is connected to in cellID

    synapse_to_cell_inidces = np.nonzero(config.SM['synapseToCell'])
    synapse = synapse_to_cell_inidces[0]
    cellID = config.SM['synapseToCell'][synapse_to_cell_inidces]

    # x is a vector of (linear) indices of active cells - note cellID contains the indices of the cells corresponding to
    # each of the synapses -- so a particular cell index will appear multiple times.
    # Thus, size of x is NOT the equal to the number of active cells,
    # but is equal to the number of synapses connected to active cells.

    x = config.SM['cellActive'][np.unravel_index(cellID, (config.SM['M'], config.SM['N']), order='F')] > 0
    synapseInput = synapse[x]

    xL = config.SM['cellLearn'][np.unravel_index(cellID, (config.SM['M'], config.SM['N']), order='F')] > 0
    synapseInputL = synapse[xL]

    aboveThresh = np.nonzero(config.SM['synapsePermanence'] > config.SM['P_thresh'])[0]
    config.SM['synapseActive'] = np.intersect1d(synapseInput, aboveThresh)

    aboveZero = np.nonzero(config.SM['synapsePermanence'] > 0)[0]
    config.SM['synapsePositive'] = np.intersect1d(synapseInput, aboveZero)

    config.SM['synapseLearn'] = np.intersect1d(synapseInputL, aboveZero)

    # Mark the active dendrites -- those with more than Theta number of active synapses
    #
    # First it computes the count of active synapses for each dendrite in SM.DendriteActive, indexed by the dendrites
    # synapseActive is the list of active synapses, i.e. synapses connected to active input and with permanence above P_thresh
    # SynapseToDendrite is an array that stores the dendrite id for each synapse
    # histogram of the array SynapseToDendrite (synapseActive) would do the job too.

    d = config.SM['synapseToDendrite'][config.SM['synapseActive']] # Dendrites with active synapses connected to active input and synapses above p_thresh
    ud = np.unique(d)
    y = np.histogram(d, ud)[0]
    config.SM['dendriteActive'][ud] = y.reshape(np.size(y), 1)
    config.SM['dendriteActive'] = (config.SM['dendriteActive'] > config.SM['Theta']).astype(int)

    # Mark the potentially active dendrites with their total

    d = config.SM['synapseToDendrite'][config.SM['synapsePositive']]
    ud = np.unique(d)
    y = np.histogram(d, ud)
    config.SM['dendritePositive'][ud] = y.reshape(np.size(y), 1)  # number of active synapses for each dendrite

    # Mark the learning dendrites with their total
    d = config.SM['synapseToDendrite'][config.SM['synapseLearn']]
    ud = np.unique(d)
    y = np.histogram(d, ud)[0]
    config.SM['dendriteLearn'][ud] = y.reshape(np.size(y), 1)
    config.SM['dendriteLearn'] = (config.SM['dendriteLearn'] > config.SM['Theta']).astype(int)

    # Mark the predicted cells as those with at least one active dendrite
    # DendriteToCell vector stores the index of the cell body it is connected to (affecting)
    # multiple dendrites can be connected to a cell body so the vector will have repeating entries of cell indices

    config.SM['cellPredicted'][:] = 0
    ind_da = np.nonzero(config.SM['dendriteActive'])
    u = np.unique(config.SM['dendriteToCell'][ind_da[0]])
    config.SM['cellPredicted'][np.unravel_index(u, np.shape(config.SM['cellPredicted']), order='F')] = 0
