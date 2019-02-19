import config
import initialize
from encoderNAB import encoderNAB

def main(inFile, outFile, displayflag, learnFlag, learntDataFile):
    config.SP['width'] = 21
    en = encoderNAB()
    en.encode(inFile, config.SP['width'])
    print("before rt")
    print(config.data)
    print("After encode function")
    initialize.initialize()
    # initialize


if __name__ == '__main__':
    inFile = "C:/Users/kamidi/Desktop/NCRG Janardhan/HTM/HTM Code/NCRG pythonHTM/NAB_input_csv_files/numentaTM_speed_7578.csv"
    main(inFile, '', False, True, '')