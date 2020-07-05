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
import secrets
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import yfinance as yf
from stock_prediction_lstm import LongShortTermMemory
from stock_prediction_numpy import StockData
from stock_prediction_plotter import Plotter

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def train_LSTM_network(start_date, ticker, validation_date):
    min_max = MinMaxScaler(feature_range=(0, 1))
    sec = yf.Ticker(ticker)
    end_date = datetime.today()
    print('End Date: ' + end_date.strftime("%Y-%m-%d"))
    data = yf.download([ticker], start=start_date, end=end_date)[['Close']]
    data = data.reset_index()
    print(data)

    plotter = Plotter(True, project_folder, sec.info['shortName'], sec.info['currency'], STOCK_TICKER)

    training_data = data[data['Date'] < validation_date].copy()
    test_data = data[data['Date'] >= validation_date].copy()
    training_data = training_data.set_index('Date')
    # Set the data frame index using column Date
    test_data = test_data.set_index('Date')
    print(test_data)
    plotter.plot_histogram_data_split(training_data, test_data, validation_date)

    data = StockData()
    (x_train, y_train), (x_test, y_test) = data.to_numpy(TIME_STEPS, min_max, training_data, test_data)

    print(x_test)

    lstm = LongShortTermMemory(project_folder)
    model = lstm.create_model(x_train)

    defined_metrics = [
        tf.keras.metrics.MeanSquaredError(name='MSE')
    ]

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=defined_metrics)
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test),
                        callbacks=[callback])
    print("saving weights")
    model.save(os.path.join(project_folder, 'model_weights.h5'))
    plotter.plot_loss(history)
    plotter.plot_mse(history)

    print("display the content of the model")
    baseline_results = model.evaluate(x_test, y_test, verbose=2)
    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)
    print()

    print("plotting prediction results")
    test_predictions_baseline = model.predict(x_test)
    test_predictions_baseline = min_max.inverse_transform(test_predictions_baseline)
    test_predictions_baseline = pd.DataFrame(test_predictions_baseline)
    test_predictions_baseline.to_csv(os.path.join(project_folder, 'predictions.csv'))

    test_predictions_baseline.rename(columns={0: STOCK_TICKER + '_predicted'}, inplace=True)
    test_predictions_baseline = test_predictions_baseline.round(decimals=0)
    test_predictions_baseline.index = test_data.index
    plotter.project_plot_predictions(test_predictions_baseline, test_data)

    print("prediction is finished")


# The Main function requires 3 major variables
# Ticker => defines the short code of a stock
# Start date => Date when we want to start using the data for training, usually the first data point of the stock
# Validation date => Date when we want to start partitioning our data from training to validation
if __name__ == '__main__':
    STOCK_TICKER = 'GOOG'
    STOCK_START_DATE = pd.to_datetime('2004-08-01')
    STOCK_VALIDATION_DATE = pd.to_datetime('2017-01-01')
    EPOCHS = 100
    BATCH_SIZE = 32
    TIME_STEPS = 60
    TODAY_RUN = datetime.today().strftime("%Y%m%d")
    TOKEN = STOCK_TICKER + '_' + TODAY_RUN + '_' + secrets.token_hex(16)
    print('Ticker: ' + STOCK_TICKER)
    print('Start Date: ' + STOCK_START_DATE.strftime("%Y-%m-%d"))
    print('Validation Date: ' + STOCK_START_DATE.strftime("%Y-%m-%d"))
    print('Generating folder: ' + TOKEN)
    # create project run folder
    project_folder = os.path.join(os.getcwd(), TOKEN)
    if not os.path.exists(project_folder):
        os.makedirs(project_folder)

    # Execute Deep Learning model
    train_LSTM_network(STOCK_START_DATE, STOCK_TICKER, STOCK_VALIDATION_DATE)
