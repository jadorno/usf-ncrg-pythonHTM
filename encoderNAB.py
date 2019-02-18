import numpy as np
import pandas as pd
import datetime
# import csv

import config


class encoderNAB:
    """Encodes the data in the input csv file provided in the Numenta Anomaly
    """
    # data = {}
    def __init__(self):
        print("encoder constructor called")

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
        config.data['fieldNames'] = np.array(['data_value', 'month', 'day_of_week','time_of_day', 'weeeknd'])
        config.data['fields'] = np.array([0, 3])
        config.data['buckets'] = np.array([120, 12, 7, 24, 2])
        config.data['width'] = np.array([width, width, width, width, width])
        config.data['circularP'] = np.array([False, True, True, True, True])
        config.data['shift'] = np.array([1, 1, 1, 1, width])

        # Read Data
        readData = pd.read_csv(filename)
        config.data['N'] = readData.shape[0]
        print(config.data['fields'])
        print(readData.ix[:, 0])
        # dateTime = readData.ix[1:readData.shape[0], 0]
        rawData = self.datevec(readData.ix[:, 0].values)
        print(rawData)
        rawData[:, 0] = readData.ix[:, 1]
        print("before printing rawData")
        print(rawData)
        print("after printing rawData")
        config.data['labels'] = readData.ix[:, 4]
        config.data['numentaAnomalyScore'] = readData.ix[:, 2]
        config.data['numentaRawAnomalyScore'] = readData.ix[:, 3]
        print(config.data)

        # Decide on bits of representation
        config.data['nBits'] = config.data['shift'] * config.data['buckets']
        config.data['nBits'][4] = 2*config.data['width'][4]
        config.data['nBits'][0] = config.data['shift'][0] * config.data['buckets'][0] + config.data['width'][0] - 1
        print("before self data print")
        print(config.data)
        print("after self data print")

        # Assign the selected data as specified in the variable data.fields to the output
        config.data['name'] = ["" for x in range(config.data['fieldNames'].shape[0])]
        config.data['value'] = np.zeros(config.data['fieldNames'].shape).tolist()
        config.data['code'] = np.zeros(config.data['fieldNames'].shape).tolist()
        for i in range(0, len(config.data['fields'])):
            j = config.data['fields'][i]
            config.data['name'][j] = config.data['fieldNames'][j]

            # Quantize data
            dataRange = (np.max(rawData[:, j]) - np.min(rawData[:, j]))
            if dataRange:
                config.data['value'][j] = np.int64(np.floor( (config.data['buckets'][j] - 1) * (rawData[:, j] - np.min(rawData[:, j])) / dataRange + 1 ))
            else:
                config.data['value'][j] = np.ones((config.data['N'], 1))
            config.data['code'][j] = self.encodeScalar(config.data['nBits'][j], config.data['buckets'][j], config.data['width'][j], config.data['shift'][j])
            print(config.data['code'][j].shape)
        print(config.data['code'])
        # return config.data

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


