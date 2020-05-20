# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly as py
import plotly.graph_objs as go
import requests
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib


def train(pair, start, end, period):
    ret = requests.get(
        'https://poloniex.com/public?command=returnChartData&currencyPair=%s&start=%s&end=%s&period=%s' % (pair, start, end, period))
    # print(ret)
    js = ret.json()
    print(js)
    df = pd.DataFrame(js)
    scaler = MinMaxScaler()
    df[['close']] = scaler.fit_transform(df[['close']])
    print(df)

    price = df['close'].values.tolist()
    window_size = 5
    x = []
    y = []

    for i in range(len(price) - window_size):
        x.append([price[i+j] for j in range(window_size)])
        y.append(price[window_size + i])
    x.append([price[len(price) - window_size + j] for j in range(window_size)])  # lastDay predict data
    print(x[-1])
    x = np.asarray(x)
    y = np.asarray(y)

    train_test_split = 1000
    x_train = x[:train_test_split, :]
    x_train = np.reshape(x_train, ( x_train.shape[0], x_train.shape[1],1))
    y_train = y[:train_test_split]
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    print(x_train.shape, y_train.shape)
    print(x_train[999])

    x_test = x[train_test_split: , :]
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
    y_test = y[train_test_split:]
    y_test = np.reshape(y_test, (y_test.shape[0], 1))

    model = Sequential()
    model.add(LSTM(128, input_shape=(5,1)))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    model.summary()

    model.fit(x_train, y_train, epochs=10, batch_size=1)

    model_dir = pair + '/' + period
    # save scaler
    joblib.dump(scaler, model_dir + 'scaler.plk')

    # save model
    model_json = model.to_json()
    with open(model_dir + "model.json", "w") as json_file :
        json_file.write(model_json)

    # save weight
    model.save_weights(model_dir + "model.h5")
    print("Saved model to disk in " + model_dir)


    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)

    print(train_predict.shape)
    print(test_predict.shape)

if __name__ == '__main__':
    plt.style.use('bmh')

    plt.figure(figsize=(10,10))
    # plt.plot(price)

    split_pt = train_test_split + window_size
    # plt.plot(np.arange(window_size, split_pt, 1), train_predict, color='g')
    # plt.plot(np.arange(split_pt, split_pt + len(test_predict), 1), test_predict, color='r')

    trace = go.Scatter(x=np.arange(1, len(price), 1), y=price, mode= 'lines' , name='original')
    trace2 = go.Scatter(x=np.arange(window_size, split_pt, 1), y=train_predict.reshape(1000),
                        mode='lines', name='train')
    trace3 = go.Scatter(x=np.arange(split_pt, split_pt + len(test_predict),1), y=test_predict.reshape(822),
                        mode='lines', name='pred')
    data = [trace, trace2, trace3]
    py.offline.plot(data)
