import numpy as np
import pandas as pd
import datetime
# import csv

import config

# [ToDo: Replace all data and use data from config].

class encoderNAB:
    """Encodes the data in the input csv file provided in the Numenta Anomaly
    """
    data = {}
    def __init__(self):
        print("encoder constructor called")
        # data = {}
        # self.fields
        # self.buckets
        # self.width
        # self.circularP
        # self.shift
        # self.N
        # self.labels
        # self.numentaAnomalyScore
        # self.numentaRawAnomalyScore
        # self.nBits
        # self.name
        # self.value
        # self.code

    def datevec(self, date_time):
        """Converts datetime of format '2015-09-08 11:39:00' to %[y, month, d, timeOfDay, m, s] """
        data_row = np.empty([], int)
        first = True
        for dt in date_time:
            dat = datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
            week_day = dat.weekday()
            week_end = ((dat.weekday() == 6) + ( dat.weekday() == 5)) + 1.0
            li = np.array([int(dat.year), int(dat.month), int(week_day), int(dat.hour), int(week_end), int(dat.second)])
            print(li)
            if True == first:
                data_row = np.array([li])
                first = False
            else:
                data_row = np.append(data_row, [li], axis=0)

        return data_row


    def encode(self, filename, width):
        """encode the data"""
        print("inside encode function")
        self.data['fieldNames'] = np.array(['data_value', 'month', 'day_of_week','time_of_day', 'weeeknd'])
        self.data['fields'] = np.array([0, 3])
        self.data['buckets'] = np.array([120, 12, 7, 24, 2])
        self.data['width'] = np.array([width, width, width, width, width])
        self.data['circularP'] = np.array([False, True, True, True, True])
        self.data['shift'] = np.array([1, 1, 1, 1, width])

        # Read Data
        readData = pd.read_csv(filename)
        self.data['N'] = readData.shape[0]
        print(self.data['fields'])
        print(readData.ix[:, 0])
        # dateTime = readData.ix[1:readData.shape[0], 0]
        rawData = self.datevec(readData.ix[:, 0].values)
        print(rawData)
        rawData[:, 0] = readData.ix[:, 1]
        print("before printing rawData")
        print(rawData)
        print("after printing rawData")
        self.data['labels'] = readData.ix[:, 4]
        self.data['numentaAnomalyScore'] = readData.ix[:, 2]
        self.data['numentaRawAnomalyScore'] = readData.ix[:, 3]
        print(self.data)

        # Decide on bits of representation
        self.data['nBits'] = self.data['shift'] * self.data['buckets']
        self.data['nBits'][4] = 2*self.data['width'][4]
        self.data['nBits'][0] = self.data['shift'][0] * self.data['buckets'][0] + self.data['width'][0] - 1
        print("before self data print")
        print(self.data)
        print("after self data print")

        # Assign the selected data as specified in the variable data.fields to the output
        self.data['name'] = ["" for x in range(self.data['fieldNames'].shape[0])]
        self.data['value'] = np.zeros(self.data['fieldNames'].shape).tolist()
        self.data['code'] = np.zeros(self.data['fieldNames'].shape).tolist()
        for i in range(0, len(self.data['fields'])):
            j = self.data['fields'][i]
            self.data['name'][j] = self.data['fieldNames'][j]

            # Quantize data
            dataRange = (np.max(rawData[:, j]) - np.min(rawData[:, j]))
            if dataRange:
                self.data['value'][j] = np.int64(np.floor( (self.data['buckets'][j] - 1) * (rawData[:, j] - np.min(rawData[:, j])) / dataRange + 1 ))
            else:
                self.data['value'][j] = np.ones((self.data['N'], 1))
            self.data['code'][j] = self.encodeScalar(self.data['nBits'][j], self.data['buckets'][j], self.data['width'][j], self.data['shift'][j])
            print(self.data['code'][j].shape)
        print(self.data['code'])
        return self.data

    def encodeScalar(self, n, buckets, width, shift):
        """Returns the SCR"""
        rand = 200
        scr = np.append(np.ones(width, int), np.zeros(n-width, int), axis=0)
        final_scr = np.array([scr])
        for i in range(1, buckets):
            scr = np.roll(scr, 1)
            final_scr = np.append(final_scr, [scr], axis=0)
        print(final_scr.shape)
        print(final_scr)
        return final_scr


