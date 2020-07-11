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
from absl import app
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from stock_prediction_class import StockPrediction
from stock_prediction_numpy import StockData
from datetime import date, timedelta
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


def main(argv):
    print(tf.version.VERSION)
    inference_folder = os.path.join(os.getcwd(), RUN_FOLDER)
    stock = StockPrediction(STOCK_TICKER, STOCK_START_DATE, STOCK_VALIDATION_DATE, inference_folder)

    data = StockData(stock)

    (x_train, y_train), (x_test, y_test), (training_data, test_data) = data.download_transform_to_numpy(TIME_STEPS, inference_folder)
    min_max = data.get_min_max()

    # load future data

    print('Latest Stock Price')
    latest_close_price = test_data.Close.iloc[-1]
    latest_date = test_data[-1:]['Close'].idxmin()
    print(latest_close_price)
    print('Latest Date')
    print(latest_date)

    tomorrow_date = latest_date + timedelta(1)
    next_year = latest_date + timedelta(TIME_STEPS*30)

    print('Future Date')
    print(tomorrow_date)

    print('Future Timespan Date')
    print(next_year)

    x_test, y_test, test_data = data.generate_future_data(TIME_STEPS, min_max, tomorrow_date, next_year, latest_close_price)

    # load the weights from our best model
    model = tf.keras.models.load_model(os.path.join(inference_folder, 'model_weights.h5'))
    model.summary()

    #print(x_test)
    #print(test_data)
    # display the content of the model
    #baseline_results = model.evaluate(x_test, y_test, verbose=2)
    #for name, value in zip(model.metrics_names, baseline_results):
    #    print(name, ': ', value)
    #print()

    # perform a prediction
    test_predictions_baseline = model.predict(x_test)
    test_predictions_baseline = min_max.inverse_transform(test_predictions_baseline)
    test_predictions_baseline = pd.DataFrame(test_predictions_baseline)

    test_predictions_baseline.rename(columns={0: STOCK_TICKER + '_predicted'}, inplace=True)
    test_predictions_baseline = test_predictions_baseline.round(decimals=0)

    #print(test_data)
    #print(test_predictions_baseline)

    #test_predictions_baseline.index = test_data.index
    test_data.to_csv(os.path.join(inference_folder, 'generated.csv'))
    test_predictions_baseline.to_csv(os.path.join(inference_folder, 'inference.csv'))


    print("plotting predictions")
    plt.figure(figsize=(14, 5))
    plt.plot(test_predictions_baseline[STOCK_TICKER + '_predicted'], color='red', label='Predicted [' + 'GOOG' + '] price')
    plt.xlabel('Time')
    plt.ylabel('Price [' + 'USD' + ']')
    plt.legend()
    plt.title('Prediction')
    plt.savefig(os.path.join(inference_folder, STOCK_TICKER + '_future_prediction.png'))
    plt.pause(0.001)

    plt.figure(figsize=(14, 5))
    plt.plot(test_data.Close, color='green', label='Simulated [' + 'GOOG' + '] price')
    plt.xlabel('Time')
    plt.ylabel('Price [' + 'USD' + ']')
    plt.legend()
    plt.title('Random')
    plt.savefig(os.path.join(inference_folder, STOCK_TICKER + '_future_random.png'))
    plt.pause(0.001)
    plt.show()


if __name__ == '__main__':
    TIME_STEPS = 10
    RUN_FOLDER = 'GOOG_20200711_23787967bfadc708e9b507740b30b411'
    STOCK_TICKER = 'GOOG'
    STOCK_START_DATE = pd.to_datetime('2004-08-01')
    STOCK_VALIDATION_DATE = pd.to_datetime('2017-01-01')
    app.run(main)