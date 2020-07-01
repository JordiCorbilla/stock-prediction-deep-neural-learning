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

def train_LSTM_network(start_date, ticker, validation_date):
    sec = yf.Ticker("GOOG")
    data = yf.download(ticker, start=start_date, end=datetime.date.today())[['Close']]
    data = data.reset_index()
    print(data)

    training_data = data[data['Date'] < validation_date].copy()
    test_data = data[data['Date'] >= validation_date].copy()
    training_data = training_data.set_index('Date')
    test_data = test_data.set_index('Date')
    plt.figure(figsize=(12, 5))
    plt.plot(training_data.Close, color='green')
    plt.plot(test_data.Close, color='red')
    plt.ylabel("Price")
    plt.xlabel("Date")
    plt.legend(["Training Data", "Validation Data >= " + validation_date.strftime("%Y-%m-%d")])
    plt.title(sec.info['shortName'])
    training_data.hist()
    plt.show()

if __name__ == '__main__':
    start_date = pd.to_datetime('2004-08-01')
    ticker = ['GOOG']
    validation_date = pd.to_datetime('2017-01-01')
    train_LSTM_network(start_date, ticker, validation_date)