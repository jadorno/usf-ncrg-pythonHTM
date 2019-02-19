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

    # Compute overlap



