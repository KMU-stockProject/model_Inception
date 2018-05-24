import os
import csv
import pickle
import numpy as np


class Preprocessing(object):
    def __init__(self, predictRange=1):
        self.current_dir = os.getcwd()
        self.predictRange = predictRange*5
        self.predictStartPos = 46   # +1

        self.dataPath = os.path.join(self.current_dir, 'data', 'rawData')
        self.savePath = os.path.join(self.current_dir, 'data', 'pklData')

        self.linear_x = [i for i in range(1, 6)]

    def selectData(self):
        dataPath = os.path.join(self.current_dir, 'data', 'rawData')
        dataList = os.listdir(dataPath)

        for data in dataList:
            temp = []
            fp = open(os.path.join(dataPath, data))

            c = csv.reader(fp)
            for r in c:
                temp.append(r)

            if int(temp[0][0]) > 20071214:  # only 9years or more
                os.remove(os.path.join(dataPath, data))

            fp.close()

    def temp(self, d1, d2):
        if d2 == 0:
            return 0 if d1 is 0 else d1*100

        return ((d1-d2)/d2)*100

    # 0: date, 1: start, 2: high, 3: low, 4: close, 5: volume
    def MA(self, data, pos):
        temp = [i[pos] for i in data]
        return sum(temp) / len(temp)

    def WMA(self, data, pos):
        temp = [(i+1)*data[i][pos] for i in range(len(data))]
        return sum(temp) / len(temp)

    def linear(self, data):
        y = [data[i][4] for i in range(5)]
        return np.polyfit(self.linear_x, y, 1)[0]

    def makeSimpleData(self, rawData, isTrain):
        rawData = rawData[isTrain]
        data = list()
        print(rawData[self.predictStartPos - 1])
        for i in range(self.predictStartPos, len(rawData) - self.predictRange):
            temp = list()
            for j in range(1, 27):
                pos = i - j
                temp.insert(0, [self.temp(rawData[pos][1], rawData[pos - 1][1]),
                                self.temp(rawData[pos][2], rawData[pos - 1][2]),
                                self.temp(rawData[pos][3], rawData[pos - 1][3]),
                                self.temp(rawData[pos][4], rawData[pos - 1][4]),
                                self.temp(rawData[pos][5], rawData[pos - 1][5])])

            data.append(temp)
        return data


    def makeMVData(self, rawData, isTrain):
        rawData = rawData[isTrain]
        data1 = list()
        data2 = list()
        print(rawData[self.predictStartPos - 1:self.predictStartPos])
        for i in range(self.predictStartPos, len(rawData) - self.predictRange):
            temp1 = list()
            temp2 = list()
            for j in range(26):
                pos = i - j
                temp1.insert(0, [self.temp(self.MA(rawData[pos - 5:pos], 4), self.MA(rawData[pos - 6:pos - 1], 4)),
                                 self.temp(self.MA(rawData[pos - 10:pos], 4), self.MA(rawData[pos - 11:pos - 1], 4)),
                                 self.temp(self.MA(rawData[pos - 15:pos], 4), self.MA(rawData[pos - 16:pos - 1], 4)),
                                 self.temp(self.MA(rawData[pos - 20:pos], 4), self.MA(rawData[pos - 21:pos - 1], 4))])

                temp2.insert(0, [self.temp(self.MA(rawData[pos - 5:pos], 5), self.MA(rawData[pos - 6:pos - 1], 5)),
                                 self.temp(self.MA(rawData[pos - 10:pos], 5), self.MA(rawData[pos - 11:pos - 1], 5)),
                                 self.temp(self.MA(rawData[pos - 15:pos], 5), self.MA(rawData[pos - 16:pos - 1], 5)),
                                 self.temp(self.MA(rawData[pos - 20:pos], 5), self.MA(rawData[pos - 21:pos - 1], 5))])

            data1.append(temp1)
            data2.append(temp2)

        return [data1, data2]   # close, volume


    def makeWMVData(self, rawData, isTrain):
        rawData = rawData[isTrain]
        data1 = list()
        data2 = list()
        print(rawData[self.predictStartPos - 1:self.predictStartPos])
        for i in range(self.predictStartPos, len(rawData) - self.predictRange):
            temp1 = list()
            temp2 = list()
            for j in range(26):
                pos = i - j
                temp1.insert(0, [self.temp(self.WMA(rawData[pos - 5:pos], 4), self.WMA(rawData[pos - 6:pos - 1], 4)),
                                 self.temp(self.WMA(rawData[pos - 10:pos], 4), self.WMA(rawData[pos - 11:pos - 1], 4)),
                                 self.temp(self.WMA(rawData[pos - 15:pos], 4), self.WMA(rawData[pos - 16:pos - 1], 4)),
                                 self.temp(self.WMA(rawData[pos - 20:pos], 4), self.WMA(rawData[pos - 21:pos - 1], 4))])

                temp2.insert(0, [self.temp(self.WMA(rawData[pos - 5:pos], 5), self.WMA(rawData[pos - 6:pos - 1], 5)),
                                 self.temp(self.WMA(rawData[pos - 10:pos], 5), self.WMA(rawData[pos - 11:pos - 1], 5)),
                                 self.temp(self.WMA(rawData[pos - 15:pos], 5), self.WMA(rawData[pos - 16:pos - 1], 5)),
                                 self.temp(self.WMA(rawData[pos - 20:pos], 5), self.WMA(rawData[pos - 21:pos - 1], 5))])

            data1.append(temp1)
            data2.append(temp2)

        return [data1, data2] # close, volume


    def makeLinearData(self, rawData, isTrain):
        rawData = rawData[isTrain]
        data = list()
        print(rawData[self.predictStartPos - 1:self.predictStartPos])
        for i in range(self.predictStartPos, len(rawData) - self.predictRange):
            temp = list()
            for j in range(26):
                pos = i - j
                temp.insert(0, [self.temp(self.linear(rawData[pos - 5:pos]), self.linear(rawData[pos - 8:pos - 3])),
                                self.temp(self.linear(rawData[pos - 5:pos]), self.linear(rawData[pos - 10:pos - 5])),
                                self.temp(self.linear(rawData[pos - 5:pos]), self.linear(rawData[pos - 13:pos - 8])),
                                self.temp(self.linear(rawData[pos - 5:pos]), self.linear(rawData[pos - 18:pos - 13]))])

            data.append(temp)
        return data


    def makeTargetData(self, rawData, isTrain):
        rawData = rawData[isTrain]
        data = list()
        print(rawData[self.predictStartPos - 1])
        for i in range(self.predictStartPos, len(rawData) - self.predictRange):
            r = self.temp(rawData[i + self.predictRange - 1][4], rawData[i - 1][4])
            if r < -3.0:
                data.append([1.0, 0.0, 0.0, 0.0])
            elif r < 0:
                data.append([0.0, 1.0, 0.0, 0.0])
            elif r <= 3.0:
                data.append([0.0, 0.0, 1.0, 0.0])
            else:
                data.append([0.0, 0.0, 0.0, 1.0])

        return data


    def preprocessing(self):
        dataList = os.listdir(self.dataPath)

        for data in dataList:
            fp = open(os.path.join(self.dataPath, data))
            fileName = data.split('.')[0]

            csvData = csv.reader(fp)
            temp = list()
            for i in csvData:
                t = [float(j) for j in i]
                temp.append(t)

            trainEnd, testStart = int(len(temp) * 0.7), int(len(temp) * 0.69)
            rawData = [temp[testStart:], temp[:trainEnd]]
            print(fileName)
            dir = {True: 'training', False: 'test'}
            for isTrain in [True, False]:
                # simple, closeMV, volumeMV, closeWMV, volumeWMV, linear, target
                temp = list()
                temp.append(self.makeSimpleData(rawData, isTrain))
                temp.extend(self.makeMVData(rawData, isTrain))
                temp.extend(self.makeWMVData(rawData, isTrain))
                temp.append(self.makeLinearData(rawData, isTrain))
                temp.append(self.makeTargetData(rawData, isTrain))

                print(len(temp[0]) == len(temp[1]), len(temp[1]) == len(temp[2]), len(temp[2]) == len(temp[3]),
                      len(temp[3]) == len(temp[4]), len(temp[4]) == len(temp[5]), len(temp[5]))
                fp1 = open(os.path.join(self.savePath, dir[isTrain], '{}.pkl').format(fileName), 'wb')
                pickle.dump(temp, fp1)

                fp1.close()
                fp.close()