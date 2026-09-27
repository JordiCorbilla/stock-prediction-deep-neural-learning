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
import warnings

warnings.filterwarnings("ignore", message=".*np.object.*", category=FutureWarning)

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Add, Concatenate, Dense, Dropout, Input, LSTM, Subtract
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
        if version in ('v4', 'v8', 'v9'):
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
        if version in ('v5', 'v6'):
            return self._create_model_v5(x_train, output_units)
        if version == 'v7':
            return self._create_model_v7(x_train, output_units)
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

    def _create_model_v7(self, x_train, output_units, activation=None):
        model = Sequential()
        model.add(Input(shape=(x_train.shape[1], x_train.shape[2])))
        model.add(LSTM(units=128, return_sequences=True))
        model.add(Dropout(0.1))
        model.add(LSTM(units=64))
        model.add(Dropout(0.2))
        model.add(Dense(units=output_units, activation=activation))
        model.summary()
        return model

    def create_multitask_model(self, x_train):
        inputs = Input(shape=(x_train.shape[1], x_train.shape[2]), name='market_window')
        shared = LSTM(units=128, return_sequences=True, name='shared_lstm_1')(inputs)
        shared = Dropout(0.1, name='shared_dropout_1')(shared)
        shared = LSTM(units=64, name='shared_lstm_2')(shared)
        shared = Dropout(0.2, name='shared_dropout_2')(shared)

        direction = Dense(units=1, activation='sigmoid', name='direction')(shared)
        magnitude = Dense(units=1, activation='softplus', name='magnitude')(shared)

        model = tf.keras.Model(
            inputs=inputs,
            outputs={'direction': direction, 'magnitude': magnitude},
            name='multitask_lstm_v8',
        )
        model.summary()
        return model

    def create_return_multitask_model(self, x_train):
        """Create v9: shared LSTM encoder with return, direction and ordered quantile heads."""
        inputs = Input(shape=(x_train.shape[1], x_train.shape[2]), name='market_window')
        shared = LSTM(units=128, return_sequences=True, name='shared_lstm_1')(inputs)
        shared = Dropout(0.1, name='shared_dropout_1')(shared)
        shared = LSTM(units=64, name='shared_lstm_2')(shared)
        shared = Dropout(0.2, name='shared_dropout_2')(shared)

        direction = Dense(units=1, activation='sigmoid', name='direction')(shared)
        expected_return = Dense(units=1, activation='linear', name='expected_return')(shared)

        median = Dense(units=1, activation='linear', name='median_return')(shared)
        lower_distance = Dense(units=1, activation='softplus', name='lower_distance')(shared)
        upper_distance = Dense(units=1, activation='softplus', name='upper_distance')(shared)
        q10 = Subtract(name='q10')([median, lower_distance])
        q90 = Add(name='q90')([median, upper_distance])
        quantiles = Concatenate(name='quantiles')([q10, median, q90])

        model = tf.keras.Model(
            inputs=inputs,
            outputs={
                'direction': direction,
                'expected_return': expected_return,
                'quantiles': quantiles,
            },
            name='return_multitask_lstm_v9',
        )
        model.summary()
        return model

    def get_loss(self, version='v1'):
        if version in ('v2', 'v3', 'v4', 'v5'):
            return Huber()
        return 'mean_squared_error'

    def get_optimizer(self, version='v1'):
        if version in ('v4', 'v5', 'v9'):
            return Adam(learning_rate=0.001)
        return 'adam'
