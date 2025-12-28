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
import warnings

warnings.filterwarnings("ignore", message=".*np.object.*", category=FutureWarning)

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, Dense, LSTM, Input
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam


class LongShortTermMemory:
    def __init__(self, project_folder):
        self.project_folder = project_folder

    def get_defined_metrics(self):
        defined_metrics = [
            tf.keras.metrics.MeanSquaredError(name='MSE')
        ]
        return defined_metrics

    def get_callback(self):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)
        return callback

    def get_callbacks(self, version='v1'):
        callbacks = [self.get_callback()]
        if version == 'v4':
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-5,
                    verbose=1,
                )
            )
        return callbacks

    def create_model(self, x_train, version='v1', output_units=1):
        if version in ('v2', 'v3'):
            return self._create_model_v2(x_train)
        if version == 'v4':
            return self._create_model_v4(x_train)
        if version == 'v5':
            return self._create_model_v5(x_train, output_units)
        return self._create_model_v1(x_train)

    def _create_model_v1(self, x_train):
        model = Sequential()
        model.add(Input(shape=(x_train.shape[1], x_train.shape[2])))
        model.add(LSTM(units=100, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(units=50))
        model.add(Dropout(0.5))
        model.add(Dense(units=1))
        model.summary()
        return model

    def _create_model_v2(self, x_train):
        model = Sequential()
        model.add(Input(shape=(x_train.shape[1], x_train.shape[2])))
        model.add(LSTM(units=64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=32))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.summary()
        return model

    def _create_model_v4(self, x_train):
        model = Sequential()
        model.add(Input(shape=(x_train.shape[1], x_train.shape[2])))
        model.add(LSTM(units=128, return_sequences=True))
        model.add(Dropout(0.1))
        model.add(LSTM(units=64))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.summary()
        return model

    def _create_model_v5(self, x_train, output_units):
        model = Sequential()
        model.add(Input(shape=(x_train.shape[1], x_train.shape[2])))
        model.add(LSTM(units=128, return_sequences=True))
        model.add(Dropout(0.1))
        model.add(LSTM(units=64))
        model.add(Dropout(0.2))
        model.add(Dense(units=output_units))
        model.summary()
        return model

    def get_loss(self, version='v1'):
        if version in ('v2', 'v3', 'v4', 'v5'):
            return Huber()
        return 'mean_squared_error'

    def get_optimizer(self, version='v1'):
        if version in ('v4', 'v5'):
            return Adam(learning_rate=0.001)
        return 'adam'
