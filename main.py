import config
from encoderNAB import encoderNAB

def main():
    filename = "C:/Users/kamidi/Desktop/NCRG Janardhan/HTM/HTM Code/NCRG pythonHTM/NAB_input_csv_files/numentaTM_speed_7578.csv"
    en = encoderNAB()
    en.encode(filename, 21)
    print("before rt")
    print(rt)
    print("After encode function")



if __name__ == '__main__':
    main()