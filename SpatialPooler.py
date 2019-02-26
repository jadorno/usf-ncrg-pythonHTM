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
        config.SP['minDutyCycle'] = 0.01 * np.max(config.SP['activeDutyCycle'])

        # Computes a moving average of how often column c has been active after inhibition.
        config.SP['activeDutyCycle'] = ((0.9 * config.SP['activeDutyCycle']) + (0.1 * active))

        # The boost value is a scalar between 1 and maxBoost. If activeDutyCyle(c)
        # is above minDutyCycle(c), the boost value is 1. The boost increases
        # linearly once the column's activeDutyCycle starts falling below its
        # minDutyCycle up to a maximum value maxBoost.
        config.SP['boost'] = np.minimum(config.SP['maxBoost'], np.fmax(1.0, config.SP['minDutyCycle']/config.SP['activeDutyCycle']).astype(int))

        inDuty = np.where(config.SP['overlapDutyCycle'] < config.SP['minDutyCycle'])
        print(inDuty)
        # config.SP['synapse'][]
        config.SP['synapse'][inDuty[0]] = config.SP['synapse'][inDuty[0]] + 0.1
        config.SP['synapse'] = config.SP['synapse'] * config.SP['connections']


    return active.T



    # Compute overlap



