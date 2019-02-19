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
    return active.T



    # Compute overlap



