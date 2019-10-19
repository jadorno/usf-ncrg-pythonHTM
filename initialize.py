import numpy as np
import math

import config

def initialize():
    """File initializes all the variables used"""
    # Parameters for spatial pooler
    # according to a paper "Effect of Spatial Pooler Initialization on Column Activity in
    # Hierarchical Temporal Memory" (1 - connectPerm < Theta/width) of input representation

    config.SP['potentialPct'] = 0.8  # Input percentage that are potential synapse.
    config.SP['connectPerm'] = 0.2  # Synapses with permanence above this are considered connected.
    config.SP['synPermActiveInc'] = 0.003  # Increment Permanence value for active synapse.
    config.SP['synPermInactiveDec'] = 0.0005  # Decrement of permanence for inactive synapse.
    config.SP['stimulusThreshold'] = 2  # Background noise level from the encoder. Usually set to very low value.
    config.SP['activeSparse'] = 0.02  # sparsity of the representation.
    config.SP['maxBoost'] = 1  # 10.


    # Parameters for Sequence Memory
    config.SM['N'] = 2048  # Number of columns N
    config.SM['M'] = 32  # Number of cells per column M
    config.SM['Nd'] = 128  # Maximum number of dendritic segments per cell
    config.SM['Ns'] = 128  # Maximum number of synapses per dendritic segment
    config.SM['Nss'] = 30  # Maximum number of synapses per dendritic segment

    config.SM['Theta'] = 20  # Dendritic segment activation threshold
    config.SM['minPositiveThreshold'] = 10  #
    config.SM['P_initial'] = 0.24  # Initial synaptic permanence
    config.SM['P_thresh'] = 0.5  # Connection threshold for synaptic permanence
    config.SM['P_incr'] = 0.04  # synaptic permanence increment
    config.SM['P_decr'] = 0.008  # synaptic permanence decrement
    config.SM['P_decr_pred'] = 0.001  # Synaptic permanence decrement for predicted inactive segments

    # Setup arrays for spatial pooler
    config.SP['boost'] = np.ones((config.SM['N'], 1))
    config.SP['activeDutyCycle'] = np.zeros((config.SM['N'], 1))
    config.SP['overlapDutyCycle'] = np.zeros((config.SM['N'], 1))
    config.SP['minDutyCycle'] = np.zeros((config.SM['N'], 1))

    # Initialize the spatial pooler
    iN = np.sum(config.data['nBits'][config.data['fields']])  # Number of bits in scalar encoder
    config.SP['connections'] = np.zeros((config.SM['N'], iN), dtype=bool)  # Stores the SP cell connections with the input space
    config.SP['synapse'] = np.zeros((config.SM['N'], iN))  # Stores the Synaptic permanence values
    W = int(round(config.SP['potentialPct'] * iN))

    for i in range(0, config.SM['N']):
        randPermTemplate = config.SP['connectPerm'] * np.random.rand(1, W) + 0.1
        connectIndex = np.sort(np.random.randint(iN, size=(1, W)))
        config.SP['synapse'][i][connectIndex[0]] = randPermTemplate
        config.SP['connections'][i][connectIndex[0]] = True

    ## Setup arrays for sequence memory
    # Copy of cell states used to predict
    config.SM['cellActive'] = np.zeros((config.SM['M'], config.SM['N']), dtype=int)
    config.SM['predictedActive'] = np.zeros((config.SM['M'], config.SM['N']), dtype=int)
    config.SM['cellActivePrevious'] = np.zeros((config.SM['M'], config.SM['N']), dtype=int)  # previous time

    config.SM['cellPredicted'] = np.zeros((config.SM['M'], config.SM['N']), dtype=int)
    config.SM['cellPredictedPrevious'] = np.zeros((config.SM['M'], config.SM['N']), dtype=int)

    config.SM['cellLearn'] = np.zeros((config.SM['M'], config.SM['N']), dtype=int)
    config.SM['cellLearnPrevious'] = np.zeros((config.SM['M'], config.SM['N']), dtype=int)

    config.SM['maxDendrites'] = round(config.SP['activeSparse'] * config.SM['N'] * config.SM['M'] * config.SM['Nd'])
    config.SM['maxSynapses'] = round(config.SP['activeSparse'] * config.SM['N'] * config.SM['M'] * config.SM['Nd'] * config.SM['Ns'])
    config.SM['totalDendrites'] = 0
    config.SM['totalSynapses'] = 0
    config.SM['newDendriteID'] = 0
    config.SM['newSynapseID'] = 0

    config.SM['numDendritesPerCell'] = np.zeros((config.SM['M'], config.SM['N']))  # stores number of dendrite information per cell
    config.SM['numSynapsesPerCell'] = np.zeros((config.SM['M'], config.SM['N']))  # stores number of dendrite information per cell
    config.SM['numSynapsesPerDendrite'] = np.zeros((config.SM['maxDendrites'], 1))

    config.SM['synapseToCell'] = np.zeros((config.SM['maxSynapses'], 1))
    config.SM['synapseToDendrite'] = np.zeros((config.SM['maxSynapses'], 1))
    config.SM['synapsePermanence'] = np.zeros((config.SM['maxSynapses'], 1))
    config.SM['synapseActive'] = []
    config.SM['synapsePositive'] = []
    config.SM['synapseLearn'] = []

    config.SM['dendriteToCell'] = np.zeros((config.SM['maxDendrites'], 1))
    config.SM['dendritePositive'] = np.zeros((config.SM['maxDendrites'], 1))
    config.SM['dendriteActive'] = np.zeros((config.SM['maxDendrites'], 1))
    config.SM['dendriteLearn'] = np.zeros((config.SM['maxDendrites'], 1))

    config.anomalyScores = np.zeros((config.SM['N'], 1))
