import numpy as np

import config

def spatialPooler(encodedInput, learnP, displayFlag):
    """Spatial pooler"""
    print(encodedInput.shape)
    print(encodedInput)
    # flip row vector input into column vector, if needed.
    if encodedInput.shape[0] == 1:
        encodedInput = encodedInput.T
    print(encodedInput.shape)

    overlap = np.matmul((config.SP['synapse'] > config.SP['connectPerm']), encodedInput)

    overThreshold = (overlap > config.SP['stimulusThreshold'])
    print(overThreshold.shape)

    # Computes a moving average of how often column c has overlap greater than stimulusThreshold
    config.SP['overlapDutyCycle'] = (0.9 * config.SP['overlapDutyCycle']) + (0.1 * overThreshold)

    overlap = overThreshold * overlap

    if(learnP):
        overlap = overlap * config.SP['boost']

    # Inhibit responses -- pick the top k columns
    print(overlap.shape)
    I = np.argsort(-overlap, axis=0)
    overlap[I[round(config.SP['activeSparse'] * config.SM['N']):config.SM['N']]] = 0
    print(overlap.shape)
    active = overlap > 0
    print(active.shape)
    print(active)

    if learnP:
        # Learning
        activeSynapsesIndex = np.zeros(config.SP['synapse'].shape, dtype=bool)
        encodact = (encodedInput.T > 0)
        activeSynapsesIndex[active[:, 0]] = encodact
        config.SP['synapse'][activeSynapsesIndex] = np.minimum(1.0, (config.SP['synapse'][activeSynapsesIndex] + config.SP['synPermActiveInc']) )

        inactiveSynapsesIndex = np.logical_not(activeSynapsesIndex)
        config.SP['synapse'][inactiveSynapsesIndex] = np.maximum(0, (config.SP['synapse'][inactiveSynapsesIndex] - config.SP['synPermInactiveDec']))


        # Boosting
        # The inhibition radius is the entire input

    return active.T



    # Compute overlap



