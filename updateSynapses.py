import numpy as np

import config

def updateSynapses():
    """
    This function performs Hebbian learning on the HTM array. This is the last phase of an iteration.
    This update is based on the learnState of a cell.
    For all cells with learnState of 1, synapses that were active in the previous iteration get their permanence
    counts incremented by permanenceInc. All other synapses get their permanence counts decremented by permanenceDec.

    We negatively reinforce all segments of cell with incorrect prediction. Permanence counts for synapses
    are decremented by predictedSegmentDec.
    :return:
    """
    indices = np.nonzero(config.SM['dendriteToCell'])
    dendrite = indices[0]
    cellID = config.SM['dendriteToCell'][indices]
    # create a list of synapse-dendrite pairs
    dendriteID = config.SM['synapseToDendrite'][np.nonzero(config.SM['synapseToDendrite'])]
    # create a list of synapse-preCell pairs
    indices2 = np.nonzero(config.SM['synapseToCell'])
    synapse = indices2[0]
    preCell = config.SM['synapseToCell'][indices2]

    # Step 1: Find active dendrites connected to active cells to reinforce
    reinforceDendrites = config.SM['cellLearn'][np.unravel_index(cellID, np.shape(config.SM['cellLearn']), order='F')] == 1 # Logical array of shape 'cellID'

    # Step 3: Update permanences of synapses of correctly predicted cells
    # Find the active synapses of active dendrites connected to an correctly predicted cell.
    # And then update their permanence -- boost the permanence of the ones that were predicted correctly from the previous cycle
    # and weaken the permanence of the predicted cells from pervious cycle % that are not active.
    # The boost is proportional to the total "positive" sum of the dendrite synapses. This value is "passed down" to the synapse level
    # in the following steps here. In the laast statement the synapse permanences are updated based on this dendrite level value.

    reinforceSynapses = np.isin(dendriteID, dendrite[reinforceDendrites])

    strengthenSynapses = synapse[reinforceSynapses & (config.SM['synapsePermanence'][synapse] < 1)]

    config.SM['synapsePermanence'][strengthenSynapses] = config.SM['synapsePermanence'][strengthenSynapses] + config.SM['P_incr']

    # Step 5: Demphasize synapses that predicted cells that are not in active input columns
    cp = config.SM['cellPredicted'][np.unravel_index(cellID, np.shape(config.SM['cellPredicted']), order='F')] == 1 # logical array aligned with the dendrites
    cpp = config.SM['cellActive'][np.unravel_index(cellID, np.shape(config.SM['M'], config.SM['N']), order='F')] == 0
    d = cp & cpp
    x = np.isin(dendriteID, dendrite[d])
    s = synapse[x]
    s = s[(config.SM['synapsePermanence'][s] > 0).reshape(np.shape(s))]
    config.SM['synapsePermanence'][s] = config.SM['synapsePermanence'][s] - config.SM['P_decr_pred']


