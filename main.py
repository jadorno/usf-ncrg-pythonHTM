import numpy as np

import config
import initialize
from encoderNAB import encoderNAB
import SpatialPooler
import sequenceMemory

def main(inFile, outFile, displayFlag, learnFlag, learntDataFile):
    if learnFlag:
        config.SP['width'] = 21
        en = encoderNAB()
        en.encode(inFile, config.SP['width'])
        print("before rt")
        print(config.data)
        print("After encode function")
        initialize.initialize()
        # Learning mode for spatial pooler
        # We train the spatial pooler in a separate step from sequence memory
        print('Learning sparse distributed representations using spatial pooling...')
        # Use the first 15 percent of the data (upto a maximum of 750) samples for training the spatial pooler.
        trN = min(750, round(0.15 * config.data['N']))
        for iteration in range(0, trN):
            x = np.array([])  # construct the binary vector x for each measurement from the data fields
            for i in range(0, len(config.data['fields'])):
                j = config.data['fields'][i]
                x = np.append(x, config.data['code'][j][(config.data['value'][j][iteration] - 1), :])  # The code starts form 0 so, substract 1 from scalar value.
            xSM = SpatialPooler.spatialPooler(np.reshape(x, (1, len(x))), True, False)
        print('learning sparse distributed representations using spatial pooling... Done.')

        # Setup arrays
        config.predictions = np.zeros((3, config.data['N']))
        config.SM['inputPrevious'] = np.zeros((config.SM['N'], 1))
        # [ToDo: check the datatype of inputCodes, inputSDR]
        config.data['inputCodes'] = np.array([])
        config.data['inputSDR'] = np.array([])
        config.SP['boost'] = np.ones((config.SM['N'], 1))
        # no boosting in spatial pooler as it is being run in a non-learning mode

        print('Running input of length %d through sequence memory to detect anomaly...' % (config.data['N']))

        ## Iterate through the input data and feed through the spatial pooler, sequence memory as needed.
        for iteration in range(0, config.data['N']):
            ## Run through spatial pooler (SP) without learning
            # [ToDo: check the datatype of x] => float64
            x = np.array([])  # construct the binary vector x for each measurement from the data fields
            for i in range(0, len(config.data['fields'])):
                j = config.data['fields'][i]
                x = np.append(x, config.data['code'][j][(config.data['value'][j][iteration] - 1), :])  # The code starts form 0 so, substract 1 from scalar value.
            config.SM['input'] = SpatialPooler.spatialPooler(np.reshape(x, (1, len(x))), False, displayFlag)

            # [ToDo: check the datatype of inputCodes, inputSDR] => float64
            config.data['inputCodes'] = np.append(config.data['inputCodes'], x)
            config.data['inputSDR'] = np.append(config.data['inputSDR'], config.SM['input'])

            pi = (np.sum(config.SM['cellPredicted'], axis=0)).astype(int)
            config.anomalyScores[iteration] = 1 - (np.count_nonzero( pi & config.SM['input'] ) / np.count_nonzero(config.SM['input']) )

            ## Run the input hrough sequence Memory (SM) module to compute the active cells in SM and also the predictions for the next time instant.
            sequenceMemory.sequenceMemory(learnFlag)

            # if displayFlag:
            #     # [ToDo: Finish this section]
            # else:
            #     if((iteration%100) == 0):
            #         print("\n Fraction done: %f , SM.totalDendrites: %d , SM.totalSynapses: %d " % (
            #         iteration / config.data['N'], config.SM['totalDentrites'], config.SM['totalSynapses']))

            if ((iteration % 100) == 0):
                print("\n Fraction done: %f , SM.totalDendrites: %d , SM.totalSynapses: %d " % (
                    iteration / config.data['N'], config.SM['totalDentrites'], config.SM['totalSynapses']))

            config.SM['inputPrevious'] = config.SM['input']
            config.SM['cellActivePrevious'] = config.SM['cellActive']
            config.SM['cellLearnPrevious'] = config.SM['cellLearn']

        print('\n Running input of length %d through sequence memory to detect anomaly...done' %config.data['N'])

        ## Save Data
        # [ToDo: finish save Data]


if __name__ == '__main__':
    inFile = "C:/Users/kamidi/Desktop/NCRG Janardhan/HTM/HTM Code/NCRG pythonHTM/NAB_input_csv_files/numentaTM_speed_7578.csv"
    main(inFile, '', False, True, '')