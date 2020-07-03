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
import os
import pandas as pd
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
import tensorflow as tf
from keras.losses import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import datetime
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import secrets

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

def create_long_short_term_memory_model(x_train):
    model = Sequential()
    # 1st layer with Dropout regularisation
    # * units = add 100 neurons is the dimensionality of the output space
    # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
    # * input_shape => Shape of the training dataset
    model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    # 20% of the layers will be dropped
    model.add(Dropout(0.2))
    # 2nd LSTM layer
    # * units = add 50 neurons is the dimensionality of the output space
    # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
    model.add(LSTM(units=50, return_sequences=True))
    # 20% of the layers will be dropped
    model.add(Dropout(0.2))
    # 3rd LSTM layer
    # * units = add 50 neurons is the dimensionality of the output space
    # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
    model.add(LSTM(units=50, return_sequences=True))
    # 50% of the layers will be dropped
    model.add(Dropout(0.5))
    # 4th LSTM layer
    # * units = add 50 neurons is the dimensionality of the output space
    model.add(LSTM(units=50))
    # 50% of the layers will be dropped
    model.add(Dropout(0.5))
    # Dense layer that specifies an output of one unit
    model.add(Dense(units=1))
    model.summary()
    tf.keras.utils.plot_model(model, to_file=os.path.join(project_folder, 'model_lstm.png'), show_shapes=True, show_layer_names=True)
    return model

def load_data_transform(time_steps, training_data, test_data):
    min_max = MinMaxScaler(feature_range=(0, 1))
    train_scaled = min_max.fit_transform(training_data)
    data_verification(train_scaled)

    # Training Data Transformation
    x_train = []
    y_train = []
    for i in range(time_steps, train_scaled.shape[0]):
        x_train.append(train_scaled[i - time_steps:i])
        y_train.append(train_scaled[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    total_data = pd.concat((training_data, test_data), axis=0)
    inputs = total_data[len(total_data) - len(test_data) - time_steps:]
    test_scaled = min_max.fit_transform(inputs)

    # Testing Data Transformation
    x_test = []
    y_test = []
    for i in range(time_steps, test_scaled.shape[0]):
        x_test.append(test_scaled[i - n_lags:i])
        y_test.append(test_scaled[i, 0])

    x_test, y_test = np.array(X_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return (x_train, y_train), (x_test, y_test)

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

    (x_train, y_train), (x_test, y_test) = load_data_transform(60, training_data, test_data)

    model = create_long_short_term_memory_model(x_train)

    defined_metrics = [
        tf.keras.metrics.MeanSquaredError(name='MSE')
    ]

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=defined_metrics)
    regressor.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test), callbacks=[callback])

if __name__ == '__main__':
    stock_start_date = pd.to_datetime('2004-08-01')
    stock_ticker = 'GOOG'
    token = secrets.token_hex(16)
    project_folder = os.path.join(os.getcwd(), token)
    if not os.path.exists(project_folder):
        os.makedirs(project_folder)
    stock_validation_date = pd.to_datetime('2017-01-01')
    train_LSTM_network(stock_start_date, stock_ticker, stock_validation_date)