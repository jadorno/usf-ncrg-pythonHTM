import numpy as np

import config
import markActiveStates
import markLearnStates

def sequenceMemory(learnFlag):
    """Sequence Memory"""
    markActiveStates.markActiveStates()

    """"if learnFlag:
        markLearnStates.markLearnStates()
        updateSynapses.updateSynapses()

    # predic next state
    config.SM['cellPredictedPrevious'] = config.SM['cellPredicted']
    markPredictiveStates.markPredictiveStates()"""