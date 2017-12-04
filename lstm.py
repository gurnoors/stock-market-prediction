

import pandas
import numpy
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
np = numpy

from Model import Model


def create_dataset(dataset, look_back=1):
    """convert an array of values into a dataset matrix"""
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)



class LstmModel(Model):

    def __init__(self):
        self.model = None

    def fit(self, df):
        dataframe = df
        dataset = dataframe.values
        dataset = dataset.astype('float32')

        # normalize the dataset
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = self.scaler.fit_transform(dataset)

        print 'dataset shape: ' + str(dataset.shape)
        print dataframe[:10]
        self.train = dataset[:, :]

        #
        self.data_shape = dataset.shape
        self.dataset = dataset

        # split into train and test sets
        train_size = int(len(dataset) * 0.67)
        test_size = len(dataset) - train_size
        self.train, self.test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

        # reshape into X=t and Y=t+1
        self.look_back = 3
        self.trainX, self.trainY = create_dataset(self.train, self.look_back)
        self.testX, self.testY = create_dataset(self.test, self.look_back)

        # reshape input to be [samples, time steps, features]
        self.trainX = numpy.reshape(self.trainX, (self.trainX.shape[0], 1, self.trainX.shape[1]))
        self.testX = numpy.reshape(self.testX, (self.testX.shape[0], 1, self.testX.shape[1]))

        # create and fit the LSTM network
        self.model = Sequential()
        self.model.add(LSTM(4, input_shape=(1, self.look_back)))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        # TODO: epoch 100
        self.model.fit(self.trainX, self.trainY, epochs=10, batch_size=1, verbose=2)

    def predict(self, days):
        dataset = self.dataset
        predictions = []
        for day in range(days):

            self.test = dataset[-9: , :]
            self.testX, self.testY = create_dataset(self.test, self.look_back)
            self.testX = numpy.reshape(self.testX, (self.testX.shape[0], 1, self.testX.shape[1]))

            # make predictions
            # trainPredict = self.model.predict(self.trainX)
            testPredict = self.model.predict(self.testX)


            np.append(dataset, np.reshape(testPredict[5,0], (1,1)), axis=0)


            # invert predictions
            # trainPredict = self.scaler.inverse_transform(trainPredict)
            # self.trainY = self.scaler.inverse_transform([self.trainY])
            testPredict = self.scaler.inverse_transform(testPredict)
            print (testPredict, testPredict.shape)
            predictions.append(testPredict[5, 0])
            # self.testY = self.scaler.inverse_transform([self.testY])

        print dataset.shape
        print "-------------"
        print predictions
        plt.plot(predictions)
        plt.show()
        # calculate root mean squared error
        # trainScore = math.sqrt(mean_squared_error(self.trainY[0], trainPredict[:, 0]))
        # print('Train Score: %.2f RMSE' % (trainScore))
        # testScore = math.sqrt(mean_squared_error(self.testY[0], testPredict[:, 0]))
        # print('Test Score: %.2f RMSE' % (testScore))

        # shift train predictions for plotting
        trainPredictPlot = numpy.empty(self.data_shape)
        # trainPredictPlot[:, :] = numpy.nan
        # trainPredictPlot[self.look_back:len(trainPredict) + self.look_back, :] = trainPredict
        # shift test predictions for plotting
        testPredictPlot = numpy.empty(self.data_shape)
        testPredictPlot[:, :] = numpy.nan
        # testPredictPlot[len(trainPredict) + (self.look_back * 2) + 1:len(self.dataset)-1, :] = testPredict
        # plot baseline and predictions
        plt.plot(self.dataset, 'red')
        plt.plot(self.scaler.inverse_transform(self.dataset), 'orange')
        plt.plot(self.scaler.inverse_transform(dataset), 'green')
        # plt.plot(trainPredictPlot, 'blue')
        # plt.plot(testPredictPlot, 'green')
        plt.show()

def main():
    DATA_DIR = '/Users/gurnoorsinghbhatia/Documents/code/cmpe255/project/data/'
    # fix random seed for reproducibility
    numpy.random.seed(7)
    # load the dataset
    dataframe = read_csv(DATA_DIR + 'BSE_BOM539434.csv', usecols=[1], engine='python')
    model = LstmModel()
    model.fit(df=dataframe)
    model.predict(days=50)

if __name__ == '__main__':
    main()