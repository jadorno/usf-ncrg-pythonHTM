import numpy as np

import config
import initialize
from encoderNAB import encoderNAB
import SpatialPooler

def main(inFile, outFile, displayflag, learnFlag, learntDataFile):
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
                x = np.append(x, config.data['code'][j][config.data['value'][j][iteration], :])
            xSM = SpatialPooler.spatialPooler(np.reshape(x, (1, len(x))), True, False)
        print('learning sparse distributed representations using spatial pooling... Done.')







if __name__ == '__main__':
    inFile = "C:/Users/kamidi/Desktop/NCRG Janardhan/HTM/HTM Code/NCRG pythonHTM/NAB_input_csv_files/numentaTM_speed_7578.csv"
    main(inFile, '', False, True, '')