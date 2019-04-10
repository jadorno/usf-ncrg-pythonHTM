import numpy as np

import config

def markActiveStates():
    """Given the input (SM.input), this function (i) computes the active cells in the sequence memory array,
    basically, all predicted cell in an active column (column with 1 as input) is active.
    and all cells in an active column with no predicted cells are active; (ii) marks the appropriate outputs
    an anomaly score.

    It assumes that predictive states have been updated in the previoius iteration
    The next function that should be run updates the learn states -- compute_learn_states.

    The anomalyScore computation is based on description at
    https://github.com/numenta/nupic/wiki/Anomaly-Detection-and-Anomaly-Scores
    using actual predictions rather than "confidences" as stated on the
    website. (NOTE: I am not sure what column "confidences" mean -- FUTURE MODIFICATION)

    We follow the implementation that is sketched out at
    http://numenta.com/assets/pdf/biological-and-machine-intelligence/0.4/BaMI-Temporal-Memory.pdf

    The following, that is in the NUPIC, has NOT been implemented
    http://chetansurpur.com/slides/2014/5/4/cla-in-nupic.html#42
    "At the beginning of a sequence we stay in "Pay Attention Mode" for a number of timesteps (relevant parameter: pamLength)
    When we are in PAM, we do not burst unpredicted columns during learning
    If new sequence, turn on "start cell" (the first one) for every active column

    1.  If new sequence, turn on "start cell" (the first one) for every active column
    2.  Otherwise, turn on any predicted cells in every active column
    3.  If no predicted cells in a column, turn on every cell in the column"
    """

    # Find Index (row and col) of the predicted cells, i.e. cells in polarized state as stored in the
    # 2D sparse array SM.CellPredicted.
    #[ToDo: Remove commented code after testing]
    # temp_indexs = np.nonzero(config.SM['cellPredicted'])
    # rowPredicted = temp_indexs[0]
    # columnPredicted = temp_indexs[1]

    predicted_indices = config.SM['cellPredicted'] > 0

    # Find the index of the active input columns
    # columnInput = np.nonzero(config.SM['input'])[1]  # Assuming input is a vector
    columnInput = config.SM['input'] > 0

    ## Reset active cell array and the array that keeps track of the cells whose dendrites will be reinforced during learning
    config.SM['cellActive'][:] = 0
    config.SM['predictedActive'][:] = 0
    ## Set correctly predicted cells to active state

    # finds which of the predicted columns are active input column. sets all other column cell indecices to false.
    correctCells = np.copy(predicted_indices)
    correctCells[:, ~columnInput[0]] = False
    config.SM['cellActive'][correctCells] = 1
    config.SM['predictedActive'][correctCells] = 1  # needed for temporal pooling

    correctColumns = (np.sum(correctCells, axis=0) > 0)

    # Compute anomaly score - differences of the ones in the input and the correctly predicted ones
    # THIS CAN BE EXPERIMENTED WITH AND UPDATED
    anomalyScore = (1 - (np.count_nonzero(correctColumns) / np.count_nonzero(columnInput)))

    # The following uses the tip at http://floybix.github.io/2016/07/01/attempting-nab
    # Instead of the raw bursting rate, a delta anomaly score was calculated:
    # it considers only newly active columns (ignoring any remaining active from the previous timestep).
    # The bursting rate is calculated only within these new columns. To handle small changes,
    # the number of columns considered ? i.e. divided by ? is kept from falling below 20# of the total number
    # of active columns (20# of 40 = 8).
    # (number-of-newly-active-columns-that-are-bursting) /
    # max(0.2 * number-of-active-columns, number-of-newly-active-columns)
    #
    # newlyActiveColumns = setdiff (find(SM.InputPrevious), columnInput);
    #
    # newlyActiveColumnsCorrect = intersect(uniqueCorrectColumns, newlyActiveColumns);
    #
    # anomalyScore = (length(newlyActiveColumns) - length (newlyActiveColumnsCorrect))/...
    #     max (0.2*length(columnInput), length(newlyActiveColumns));
    ## Tag wrongly predicted cells -- used during learning.

    ## Burst the cells in the columns with no prediction but active input

    config.SM['burstColumns'] = (columnInput ^ correctColumns)
    config.SM['cellActive'][:, config.SM['burstColumns'][0]] = 1
