# Copyright 2020-2026 Jordi Corbilla. All Rights Reserved.
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

import numpy as np
from datetime import timedelta
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import yfinance as yf


class StockData:
    def __init__(self, stock):
        self._stock = stock
        self._sec = yf.Ticker(self._stock.get_ticker())
        self._min_max = MinMaxScaler(feature_range=(0, 1))
        self._input_scaler = MinMaxScaler(feature_range=(0, 1))

    def __data_verification(self, train):
        print('mean:', train.mean(axis=0))
        print('max', train.max())
        print('min', train.min())
        print('Std dev:', train.std(axis=0))

    def get_stock_short_name(self):
        return self._sec.info['shortName']

    def get_min_max(self):
        return self._min_max

    def get_input_scaler(self):
        return self._input_scaler

    def get_stock_currency(self):
        return self._sec.info['currency']

    def _compute_log_returns(self, series):
        return np.log(series).diff().dropna()

    def _compute_deltas(self, series):
        return series.diff().dropna()

    def _ensure_series(self, series_or_frame):
        if isinstance(series_or_frame, pd.DataFrame):
            return series_or_frame.iloc[:, 0]
        return series_or_frame

    def download_raw_data(self, end_date=None):
        if end_date is None:
            end_date = datetime.today()
        data = yf.download(self._stock.get_ticker(), start=self._stock.get_start_date(), end=end_date, progress=False, auto_adjust=False)[['Close']]
        data = data.reset_index()
        data = data.set_index('Date')
        return data

    def download_transform_to_numpy(self, time_steps, project_folder, use_returns=False, use_deltas=False):
        end_date = datetime.today()
        print('End Date: ' + end_date.strftime("%Y-%m-%d"))
        data = yf.download(self._stock.get_ticker(), start=self._stock.get_start_date(), end=end_date, progress=False, auto_adjust=False)[['Close']]
        data = data.reset_index()
        data.to_csv(os.path.join(project_folder, 'downloaded_data_'+self._stock.get_ticker()+'.csv'))
        #print(data)

        training_data = data[data['Date'] < self._stock.get_validation_date()].copy()
        test_data = data[data['Date'] >= self._stock.get_validation_date()].copy()
        training_data = training_data.set_index('Date')
        # Set the data frame index using column Date
        test_data = test_data.set_index('Date')
        #print(test_data)

        if use_returns and use_deltas:
            raise ValueError('use_returns and use_deltas cannot both be true')

        if use_returns:
            full_series = data.set_index('Date')[['Close']]
            full_series = self._ensure_series(full_series)
            returns = self._compute_log_returns(full_series).rename('Close')
            training_returns = returns[returns.index < self._stock.get_validation_date()]
            test_returns = returns[returns.index >= self._stock.get_validation_date()]
            train_scaled = self._min_max.fit_transform(training_returns.to_frame())
        elif use_deltas:
            full_series = data.set_index('Date')[['Close']]
            full_series = self._ensure_series(full_series)
            deltas = self._compute_deltas(full_series).rename('Close')
            training_deltas = deltas[deltas.index < self._stock.get_validation_date()]
            test_deltas = deltas[deltas.index >= self._stock.get_validation_date()]
            close_scaled = self._input_scaler.fit_transform(training_data)
            delta_scaled = self._min_max.fit_transform(training_deltas.to_frame())
            train_scaled = close_scaled
        else:
            train_scaled = self._min_max.fit_transform(training_data)
        self.__data_verification(train_scaled)

        # Training Data Transformation
        x_train = []
        y_train = []
        if use_deltas:
            close_scaled_aligned = close_scaled[1:]
            for i in range(time_steps, close_scaled_aligned.shape[0]):
                x_train.append(close_scaled_aligned[i - time_steps:i])
                y_train.append(delta_scaled[i, 0])
        else:
            for i in range(time_steps, train_scaled.shape[0]):
                x_train.append(train_scaled[i - time_steps:i])
                y_train.append(train_scaled[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        if use_returns:
            total_returns = pd.concat((training_returns, test_returns), axis=0)
            inputs = total_returns[len(total_returns) - len(test_returns) - time_steps:]
            test_scaled = self._min_max.transform(inputs.to_frame())
        elif use_deltas:
            total_data = pd.concat((training_data, test_data), axis=0)
            total_close_scaled = self._input_scaler.transform(total_data)
            total_close_aligned = total_close_scaled[1:]

            total_deltas = pd.concat((training_deltas, test_deltas), axis=0)
            total_deltas_scaled = self._min_max.transform(total_deltas.to_frame())

            test_start = len(training_deltas)
            inputs_x = total_close_aligned[test_start - time_steps:]
            inputs_y = total_deltas_scaled[test_start - time_steps:]
        else:
            total_data = pd.concat((training_data, test_data), axis=0)
            inputs = total_data[len(total_data) - len(test_data) - time_steps:]
            test_scaled = self._min_max.transform(inputs)

        # Testing Data Transformation
        x_test = []
        y_test = []
        if use_deltas:
            for i in range(time_steps, inputs_x.shape[0]):
                x_test.append(inputs_x[i - time_steps:i])
                y_test.append(inputs_y[i, 0])
        else:
            for i in range(time_steps, test_scaled.shape[0]):
                x_test.append(test_scaled[i - time_steps:i])
                y_test.append(test_scaled[i, 0])

        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return (x_train, y_train), (x_test, y_test), (training_data, test_data)

    def __date_range(self, start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    def negative_positive_random(self):
        return 1 if random.random() < 0.5 else -1

    def pseudo_random(self):
        return random.uniform(0.01, 0.03)

    def generate_future_data(self, time_steps, min_max, start_date, end_date, latest_close_price):
        x_future = []
        y_future = []

        # We need to provide a randomisation algorithm for the close price
        # This is my own implementation and it will provide a variation of the
        # close price for a +-1-3% of the original value, when the value wants to go below
        # zero, it will be forced to go up.

        original_price = latest_close_price

        for single_date in self.__date_range(start_date, end_date):
            x_future.append(single_date)
            direction = self.negative_positive_random()
            random_slope = direction * (self.pseudo_random())
            #print(random_slope)
            original_price = original_price + (original_price * random_slope)
            #print(original_price)
            if original_price < 0:
                original_price = 0
            y_future.append(original_price)

        test_data = pd.DataFrame({'Date': x_future, 'Close': y_future})
        test_data = test_data.set_index('Date')

        test_scaled = min_max.fit_transform(test_data)
        x_test = []
        y_test = []
        #print(test_scaled.shape[0])
        for i in range(time_steps, test_scaled.shape[0]):
            x_test.append(test_scaled[i - time_steps:i])
            y_test.append(test_scaled[i, 0])
            #print(i - time_steps)

        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        return x_test, y_test, test_data



