# Copyright 2020 Jordi Corbilla. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import pandas as pd
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
import tensorflow as tf
from keras.losses import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import datetime
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

def plot_histogram_data_split(training, test, title, date):
    plt.figure(figsize=(12, 5))
    plt.plot(training.Close, color='green')
    plt.plot(test.Close, color='red')
    plt.ylabel("Price")
    plt.xlabel("Date")
    plt.legend(["Training Data", "Validation Data >= " + date.strftime("%Y-%m-%d")])
    plt.title(title)
    training.hist()
    plt.show()

def data_verification(train):
    print('mean:', train.mean(axis=0))
    print('max', train.max())
    print('min', train.min())
    print('Std dev:', train.std(axis=0))

def train_LSTM_network(start_date, ticker, validation_date):
    sec = yf.Ticker(ticker)
    data = yf.download([ticker], start=start_date, end=datetime.date.today())[['Close']]
    data = data.reset_index()
    print(data)

    training_data = data[data['Date'] < validation_date].copy()
    test_data = data[data['Date'] >= validation_date].copy()
    training_data = training_data.set_index('Date')
    test_data = test_data.set_index('Date')
    plot_histogram_data_split(training_data, test_data, sec.info['shortName'], validation_date)

    min_max = MinMaxScaler(feature_range=(0, 1))
    train_scaled = min_max.fit_transform(training_data)
    data_verification(train_scaled)

if __name__ == '__main__':
    stock_start_date = pd.to_datetime('2004-08-01')
    stock_ticker = 'GOOG'
    stock_validation_date = pd.to_datetime('2017-01-01')
    train_LSTM_network(stock_start_date, stock_ticker, stock_validation_date)