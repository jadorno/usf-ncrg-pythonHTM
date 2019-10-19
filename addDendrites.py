import numpy as np

import config

def addDendrites(dCells, expandDendrites, nDCells):
    """

    :param dCells:  choosen cells
    :param expandDendrites: dendrite indexes
    :param nDCells: Number of dendrites
    :return:
    """
    sCells = np.ravel_multi_index(np.nonzero(config.SM['cellLearnPrevious']), np.shape(config.SM['cellLearnPrevious']), order='F')
    if (np.size(sCells) !=0 ):
        for i in range(0, nDCells):
            dC = dCells[i][0]
            if (expandDendrites[i][0] < 0): # add new dendrite
                if (config.SM['numDendritesPerCell'][np.unravel_index(dC, (config.SM['M'], config.SM['N']), 'F')] < config.SM['Nd']):
                    config.SM['numDendritesPerCell'][np.unravel_index(dC, (config.SM['M'], cssonfig.SM['N']), 'F')] += 1
                    config.SM['dendriteToCell'][config.SM['newDendriteID']][0] = dC
                    expandDendrites[i][0] = config.SM['newDendriteID']
                    config.SM['newDendriteID'] += 1
                    config.SM['totalDendrites'] += 1

            # Expand synapses of "expandDendrites"
            # Select synapses from cells in SM.cellLearnPrevious.  -- only one per column,
            # which is ensured during the selection of the cells for 'cellLearn' in previous instance
            nNew = min(config.SM['Nss'], np.size(sCells))
            if ((config.SM['numSynpasesPerDendrite'][expandDendrites[i][0]] + nNew) < config.SM['Ns']):
                config.SM['numSynpasesPerDendrite'][expandDendrites[i][0]] += nNew
                rp = np.random.permutation(np.size(sCells)) # Random permutation vector
                sCells = sCells[rp]
                sC = sCells[0:nNew] # Ns maximum random synapses per dendrite
                randPermanence = np.multiply(2*config.SM['P_incr'], np.random.rand(np.size(sC), 1)) + config.SM['P_initial'] - config.SM['P_incr']
                newSynapses = np.arange(config.SM['newSynapseID'], (config.SM['newSynapseID'] + np.size(sC)))
                config.SM['synapseToDendrite'][newSynapses] = expandDendrites[i][0] * np.ones((np.size(newSynapses), 1), dtype=int)
                config.SM['synapsePermanence'][newSynapses] = randPermanence
                config.SM['synapseToCell'][newSynapses] = sC.reshape(np.size(sC), 1)
                config.SM['numSynapsesPerCell'][newSynapses] += 1
                config.SM['newSynapseID'] += np.size(sC)
                config.SM['totalSynapses'] += np.size(sC)
