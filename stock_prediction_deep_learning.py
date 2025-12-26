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
from datetime import datetime

warnings.filterwarnings("ignore", message=".*np.object.*", category=FutureWarning)

from stock_prediction_class import StockPrediction
from stock_prediction_lstm import LongShortTermMemory
from stock_prediction_numpy import StockData
from stock_prediction_plotter import Plotter
from stock_prediction_readme_generator import ReadmeGenerator

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def _returns_to_prices(returns, start_price):
    prices = []
    current_price = start_price
    for value in returns:
        current_price = current_price * np.exp(value)
        prices.append(current_price)
    return prices


def train_LSTM_network(stock, use_returns=False):
    data = StockData(stock)
    plotter = Plotter(True, stock.get_project_folder(), data.get_stock_short_name(), data.get_stock_currency(), stock.get_ticker())
    (x_train, y_train), (x_test, y_test), (training_data, test_data) = data.download_transform_to_numpy(
        stock.get_time_steps(),
        stock.get_project_folder(),
        use_returns=use_returns,
    )
    plotter.plot_histogram_data_split(training_data, test_data, stock.get_validation_date())
    scaler_path = os.path.join(stock.get_project_folder(), 'min_max_scaler.pkl')
    with open(scaler_path, 'wb') as scaler_file:
        pickle.dump(data.get_min_max(), scaler_file)
    config = {
        'use_returns': use_returns,
        'time_steps': stock.get_time_steps(),
        'ticker': stock.get_ticker(),
        'start_date': stock.get_start_date().strftime("%Y-%m-%d"),
        'validation_date': stock.get_validation_date().strftime("%Y-%m-%d"),
    }
    config_path = os.path.join(stock.get_project_folder(), 'model_config.json')
    with open(config_path, 'w', encoding='utf-8') as config_file:
        json.dump(config, config_file, indent=2)

    lstm = LongShortTermMemory(stock.get_project_folder())
    model = lstm.create_model(x_train)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=lstm.get_defined_metrics())
    history = model.fit(x_train, y_train, epochs=stock.get_epochs(), batch_size=stock.get_batch_size(), validation_data=(x_test, y_test),
                        callbacks=[lstm.get_callback()])
    print("saving model")
    model.save(os.path.join(stock.get_project_folder(), 'model.keras'))

    plotter.plot_loss(history)
    plotter.plot_mse(history)

    print("display the content of the model")
    baseline_results = model.evaluate(x_test, y_test, verbose=2)
    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)
    print()

    print("plotting prediction results")
    test_predictions_baseline = model.predict(x_test)
    test_predictions_baseline = data.get_min_max().inverse_transform(test_predictions_baseline).flatten()
    if use_returns:
        last_train_close = training_data['Close'].iloc[-1]
        predicted_prices = _returns_to_prices(test_predictions_baseline, last_train_close)
        predictions_df = pd.DataFrame(
            {stock.get_ticker() + '_predicted': predicted_prices},
            index=test_data.index,
        )
    else:
        predictions_df = pd.DataFrame(test_predictions_baseline, index=test_data.index)
        predictions_df.rename(columns={0: stock.get_ticker() + '_predicted'}, inplace=True)
        predictions_df = predictions_df.round(decimals=0)
    predictions_df.to_csv(os.path.join(stock.get_project_folder(), 'predictions.csv'))
    plotter.project_plot_predictions(predictions_df, test_data)

    generator = ReadmeGenerator(stock.get_github_url(), stock.get_token(), data.get_stock_short_name())
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
    parser.add_argument("-epochs", default="100")
    parser.add_argument("-batch_size", default="10")
    parser.add_argument("-time_steps", default="3")
    parser.add_argument("-github_url", default="https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning/raw/master/")
    
    args = parser.parse_args()
    
    STOCK_TICKER = args.ticker
    STOCK_START_DATE = pd.to_datetime(args.start_date)
    STOCK_VALIDATION_DATE = pd.to_datetime(args.validation_date)
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batch_size)
    TIME_STEPS = int(args.time_steps)
    TODAY_RUN = datetime.today().strftime("%Y%m%d")
    TOKEN = STOCK_TICKER + '_' + TODAY_RUN + '_' + secrets.token_hex(16)
    GITHUB_URL = args.github_url
    print('Ticker: ' + STOCK_TICKER)
    print('Start Date: ' + STOCK_START_DATE.strftime("%Y-%m-%d"))
    print('Validation Date: ' + STOCK_START_DATE.strftime("%Y-%m-%d"))
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
    train_LSTM_network(stock_prediction)
