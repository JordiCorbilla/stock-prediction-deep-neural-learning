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
import secrets
import pandas as pd
import argparse
import pickle
import numpy as np
import json
import warnings
import tensorflow as tf
from tensorflow.keras.losses import Huber
from datetime import datetime

warnings.filterwarnings("ignore", message=".*np.object.*", category=FutureWarning)

from stock_prediction_class import StockPrediction
from stock_prediction_lstm import LongShortTermMemory
from stock_prediction_numpy import StockData
from stock_prediction_plotter import Plotter
from stock_prediction_readme_generator import ReadmeGenerator
from quant_forecast_lab.config import (
    DEFAULT_MODEL_VERSION,
    DEFAULT_SEED,
    DEFAULT_USE_RETURNS,
    DEFAULT_VALIDATION_FRACTION,
    validate_model_options,
)
from quant_forecast_lab.evaluation import evaluate_price_forecast
from quant_forecast_lab.reproducibility import set_global_seed


def _returns_to_prices(returns, start_price):
    prices = []
    current_price = start_price
    for value in returns:
        current_price = current_price * np.exp(value)
        prices.append(current_price)
    return prices


def _chronological_validation_split(*arrays, fraction=DEFAULT_VALIDATION_FRACTION):
    if not 0.0 < fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5.")
    if not arrays:
        raise ValueError("At least one array is required.")
    size = len(arrays[0])
    if any(len(array) != size for array in arrays):
        raise ValueError("All training arrays must have the same length.")
    if size < 3:
        raise ValueError("At least three training samples are required for a validation split.")
    split = max(1, min(size - 1, int(size * (1.0 - fraction))))
    return tuple((array[:split], array[split:]) for array in arrays)


def train_LSTM_network(
    stock,
    use_returns=DEFAULT_USE_RETURNS,
    model_version=DEFAULT_MODEL_VERSION,
    forecast_horizon=1,
    trend_window=60,
    validation_fraction=DEFAULT_VALIDATION_FRACTION,
    seed=DEFAULT_SEED,
):
    validate_model_options(model_version, use_returns)
    set_global_seed(seed)
    use_deltas = model_version in ('v3', 'v5', 'v7', 'v8')
    use_trend_residual = model_version == 'v6'
    data = StockData(stock)
    plotter = Plotter(True, stock.get_project_folder(), data.get_stock_short_name(), data.get_stock_currency(), stock.get_ticker())
    if model_version in ('v7', 'v8'):
        (x_train, y_dir_train, y_mag_train), (x_test, y_dir_test, y_mag_test), (training_data, test_data) = data.prepare_delta_direction_data(
            stock.get_time_steps(),
            stock.get_validation_date(),
        )
    else:
        (x_train, y_train), (x_test, y_test), (training_data, test_data) = data.download_transform_to_numpy(
            stock.get_time_steps(),
            stock.get_project_folder(),
            use_returns=use_returns,
            use_deltas=use_deltas,
            use_trend_residual=use_trend_residual,
            trend_window=trend_window,
            forecast_horizon=forecast_horizon,
        )
    if model_version in ('v7', 'v8'):
        (x_fit, x_val), (y_dir_fit, y_dir_val), (y_mag_fit, y_mag_val) = _chronological_validation_split(
            x_train,
            y_dir_train,
            y_mag_train,
            fraction=validation_fraction,
        )
    else:
        (x_fit, x_val), (y_fit, y_val) = _chronological_validation_split(
            x_train,
            y_train,
            fraction=validation_fraction,
        )

    plotter.plot_histogram_data_split(training_data, test_data, stock.get_validation_date())
    scaler_path = os.path.join(stock.get_project_folder(), 'min_max_scaler.pkl')
    with open(scaler_path, 'wb') as scaler_file:
        pickle.dump(data.get_min_max(), scaler_file)
    if use_deltas or use_trend_residual:
        input_scaler_path = os.path.join(stock.get_project_folder(), 'input_scaler.pkl')
        with open(input_scaler_path, 'wb') as scaler_file:
            pickle.dump(data.get_input_scaler(), scaler_file)
    config = {
        'use_returns': use_returns,
        'use_deltas': use_deltas,
        'use_trend_residual': use_trend_residual,
        'model_version': model_version,
        'forecast_horizon': forecast_horizon,
        'trend_window': trend_window,
        'time_steps': stock.get_time_steps(),
        'ticker': stock.get_ticker(),
        'start_date': stock.get_start_date().strftime("%Y-%m-%d"),
        'validation_date': stock.get_validation_date().strftime("%Y-%m-%d"),
        'validation_fraction': validation_fraction,
        'seed': seed,
    }
    config_path = os.path.join(stock.get_project_folder(), 'model_config.json')
    with open(config_path, 'w', encoding='utf-8') as config_file:
        json.dump(config, config_file, indent=2)

    lstm = LongShortTermMemory(stock.get_project_folder())
    if model_version == 'v7':
        dir_model = lstm._create_model_v7(x_fit, output_units=1, activation='sigmoid')
        dir_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        dir_history = dir_model.fit(
            x_fit,
            y_dir_fit,
            epochs=stock.get_epochs(),
            batch_size=stock.get_batch_size(),
            validation_data=(x_val, y_dir_val),
            callbacks=lstm.get_callbacks('v4'),
        )
        mag_model = lstm._create_model_v7(x_fit, output_units=1, activation=None)
        mag_model.compile(optimizer=lstm.get_optimizer('v4'), loss=Huber(), metrics=lstm.get_defined_metrics())
        mag_history = mag_model.fit(
            x_fit,
            y_mag_fit,
            epochs=stock.get_epochs(),
            batch_size=stock.get_batch_size(),
            validation_data=(x_val, y_mag_val),
            callbacks=lstm.get_callbacks('v4'),
        )
        print("saving models")
        dir_model.save(os.path.join(stock.get_project_folder(), 'model_direction.keras'))
        mag_model.save(os.path.join(stock.get_project_folder(), 'model_magnitude.keras'))
        history = mag_history
    elif model_version == 'v8':
        multitask_model = lstm.create_multitask_model(x_fit)
        multitask_model.compile(
            optimizer=lstm.get_optimizer('v4'),
            loss={
                'direction': 'binary_crossentropy',
                'magnitude': Huber(),
            },
            metrics={
                'direction': ['accuracy'],
                'magnitude': [tf.keras.metrics.MeanSquaredError(name='MSE')],
            },
        )
        history = multitask_model.fit(
            x_fit,
            {'direction': y_dir_fit, 'magnitude': y_mag_fit},
            epochs=stock.get_epochs(),
            batch_size=stock.get_batch_size(),
            validation_data=(x_val, {'direction': y_dir_val, 'magnitude': y_mag_val}),
            callbacks=lstm.get_callbacks('v8'),
        )
        print("saving multi-task model")
        multitask_model.save(os.path.join(stock.get_project_folder(), 'model_multitask.keras'))
    else:
        output_units = forecast_horizon if model_version in ('v5', 'v6') else 1
        model = lstm.create_model(x_fit, version=model_version, output_units=output_units)
        model.compile(optimizer=lstm.get_optimizer(model_version), loss=lstm.get_loss(model_version), metrics=lstm.get_defined_metrics())
        history = model.fit(
            x_fit,
            y_fit,
            epochs=stock.get_epochs(),
            batch_size=stock.get_batch_size(),
            validation_data=(x_val, y_val),
            callbacks=lstm.get_callbacks(model_version),
        )
        print("saving model")
        model.save(os.path.join(stock.get_project_folder(), 'model.keras'))

    plotter.plot_loss(history)
    plotter.plot_mse(history)

    print("display the content of the model")
    if model_version == 'v7':
        baseline_results = mag_model.evaluate(x_test, y_mag_test, verbose=2)
        for name, value in zip(mag_model.metrics_names, baseline_results):
            print(name, ': ', value)
    elif model_version == 'v8':
        baseline_results = multitask_model.evaluate(
            x_test,
            {'direction': y_dir_test, 'magnitude': y_mag_test},
            verbose=2,
            return_dict=True,
        )
        for name, value in baseline_results.items():
            print(name, ': ', value)
    else:
        baseline_results = model.evaluate(x_test, y_test, verbose=2)
        for name, value in zip(model.metrics_names, baseline_results):
            print(name, ': ', value)
    print()

    print("plotting prediction results")
    if model_version == 'v7':
        dir_pred = dir_model.predict(x_test)
        mag_pred = mag_model.predict(x_test)
        mag_pred = data.get_min_max().inverse_transform(mag_pred).flatten()
        direction = (dir_pred.flatten() >= 0.5).astype(np.float32)
        test_predictions_baseline = mag_pred * np.where(direction > 0, 1.0, -1.0)
    elif model_version == 'v8':
        multitask_pred = multitask_model.predict(x_test)
        dir_pred = multitask_pred['direction']
        mag_pred = data.get_min_max().inverse_transform(multitask_pred['magnitude']).flatten()
        direction = (dir_pred.flatten() >= 0.5).astype(np.float32)
        test_predictions_baseline = mag_pred * np.where(direction > 0, 1.0, -1.0)
    else:
        test_predictions_baseline = model.predict(x_test)
        test_predictions_baseline = data.get_min_max().inverse_transform(test_predictions_baseline)
        if model_version not in ('v5', 'v6'):
            test_predictions_baseline = test_predictions_baseline.flatten()
    if use_returns:
        last_train_close = training_data['Close'].iloc[-1]
        predicted_prices = _returns_to_prices(test_predictions_baseline, last_train_close)
        predictions_df = pd.DataFrame({stock.get_ticker() + '_predicted': predicted_prices})
    elif use_deltas:
        last_train_close = training_data['Close'].iloc[-1]
        base_close = test_data['Close'].shift(1)
        base_close.iloc[0] = last_train_close
        if model_version == 'v5':
            step_deltas = test_predictions_baseline[:, 0]
            predicted_prices = base_close.to_numpy().flatten()[:len(step_deltas)] + step_deltas
            test_data = test_data.iloc[:len(step_deltas)]
        else:
            predicted_prices = base_close.to_numpy().flatten() + test_predictions_baseline.flatten()
        predictions_df = pd.DataFrame({stock.get_ticker() + '_predicted': predicted_prices}, index=test_data.index)
    elif use_trend_residual:
        full_series = pd.concat([training_data['Close'], test_data['Close']])
        residuals = data._compute_trend_residuals(full_series, trend_window)
        base_close = test_data['Close'].shift(1)
        base_close.iloc[0] = training_data['Close'].iloc[-1]
        if model_version == 'v6':
            step_residuals = test_predictions_baseline[:, 0]
            predicted_prices = base_close.to_numpy().flatten()[:len(step_residuals)] + step_residuals
            test_data = test_data.iloc[:len(step_residuals)]
        else:
            predicted_prices = base_close.to_numpy().flatten() + test_predictions_baseline.flatten()
        predictions_df = pd.DataFrame({stock.get_ticker() + '_predicted': predicted_prices}, index=test_data.index)
    else:
        predictions_df = pd.DataFrame(test_predictions_baseline, index=test_data.index)
        predictions_df.rename(columns={0: stock.get_ticker() + '_predicted'}, inplace=True)
        predictions_df = predictions_df.round(decimals=0)
    if len(predictions_df) != len(test_data):
        predictions_df = predictions_df.tail(len(test_data))
        test_data = test_data.tail(len(predictions_df))
    predictions_df.index = test_data.index
    predictions_df.to_csv(os.path.join(stock.get_project_folder(), 'predictions.csv'))
    plotter.project_plot_predictions(predictions_df, test_data)

    actual_values = test_data['Close']
    if isinstance(actual_values, pd.DataFrame):
        actual_values = actual_values.iloc[:, 0]
    previous_actual = actual_values.shift(1)
    previous_actual.iloc[0] = training_data['Close'].iloc[-1]
    metrics = evaluate_price_forecast(
        actual_values.to_numpy(),
        predictions_df.iloc[:, 0].to_numpy(),
        previous_actual.to_numpy(),
    )
    with open(os.path.join(stock.get_project_folder(), 'forecast_metrics.json'), 'w', encoding='utf-8') as metrics_file:
        json.dump(metrics.as_dict(), metrics_file, indent=2)
    print('Forecast metrics:', metrics.as_dict())

    generator = ReadmeGenerator(stock.get_github_url(), stock.get_project_folder(), data.get_stock_short_name())
    generator.write()

    print("prediction is finished")


# The Main function requires 3 major variables
# 1) Ticker => defines the short code of a stock
# 2) Start date => Date when we want to start using the data for training, usually the first data point of the stock
# 3) Validation date => Date when we want to start partitioning our data from training to validation
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("parsing arguments"))
    parser.add_argument("-ticker", default="^FTSE")
    parser.add_argument("-start_date", default="2017-11-01")
    parser.add_argument("-validation_date", default="2021-09-01")
    parser.add_argument("-epochs", type=int, default=100)
    parser.add_argument("-batch_size", type=int, default=10)
    parser.add_argument("-time_steps", type=int, default=3)
    parser.add_argument("-github_url", default="https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning/raw/master/")
    parser.add_argument("-use_returns", default=str(DEFAULT_USE_RETURNS).lower())
    parser.add_argument("-model_version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("-forecast_horizon", type=int, default=1)
    parser.add_argument("-trend_window", type=int, default=60)
    parser.add_argument("-validation_fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    parser.add_argument("-seed", type=int, default=DEFAULT_SEED)
    
    args = parser.parse_args()
    
    STOCK_TICKER = args.ticker
    STOCK_START_DATE = pd.to_datetime(args.start_date)
    STOCK_VALIDATION_DATE = pd.to_datetime(args.validation_date)
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    TIME_STEPS = args.time_steps
    USE_RETURNS = str(args.use_returns).lower() in ("1", "true", "yes", "y")
    MODEL_VERSION = args.model_version
    FORECAST_HORIZON = args.forecast_horizon
    TREND_WINDOW = args.trend_window
    VALIDATION_FRACTION = args.validation_fraction
    SEED = args.seed
    validate_model_options(MODEL_VERSION, USE_RETURNS)
    set_global_seed(SEED)
    TODAY_RUN = datetime.today().strftime("%Y%m%d")
    TOKEN = STOCK_TICKER + '_' + TODAY_RUN + '_' + secrets.token_hex(16)
    GITHUB_URL = args.github_url
    print('Ticker: ' + STOCK_TICKER)
    print('Start Date: ' + STOCK_START_DATE.strftime("%Y-%m-%d"))
    print('Validation Date: ' + STOCK_VALIDATION_DATE.strftime("%Y-%m-%d"))
    print('Test Run Folder: ' + TOKEN)
    # create project run folder
    PROJECT_FOLDER = os.path.join(os.getcwd(), TOKEN)
    if not os.path.exists(PROJECT_FOLDER):
        os.makedirs(PROJECT_FOLDER)

    stock_prediction = StockPrediction(STOCK_TICKER, 
                                       STOCK_START_DATE, 
                                       STOCK_VALIDATION_DATE, 
                                       PROJECT_FOLDER, 
                                       GITHUB_URL,
                                       EPOCHS,
                                       TIME_STEPS,
                                       TOKEN,
                                       BATCH_SIZE)
    # Execute Deep Learning model
    train_LSTM_network(
        stock_prediction,
        use_returns=USE_RETURNS,
        model_version=MODEL_VERSION,
        forecast_horizon=FORECAST_HORIZON,
        trend_window=TREND_WINDOW,
        validation_fraction=VALIDATION_FRACTION,
        seed=SEED,
    )
