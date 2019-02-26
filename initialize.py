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


