# Copyright 2020-2024 Jordi Corbilla. All Rights Reserved.
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
from datetime import timedelta, datetime

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def main(argv):
    print(tf.version.VERSION)
    inference_folder = os.path.join(os.getcwd(), RUN_FOLDER)
    stock = StockPrediction(STOCK_TICKER, STOCK_START_DATE, STOCK_VALIDATION_DATE, inference_folder, GITHUB_URL, EPOCHS, TIME_STEPS, TOKEN, BATCH_SIZE)

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
    # Specify the next 300 days
    next_year = latest_date + timedelta(TIME_STEPS * 100)

    print('Future Date')
    print(tomorrow_date)

    print('Future Timespan Date')
    print(next_year)

    x_test, y_test, test_data = data.generate_future_data(TIME_STEPS, min_max, tomorrow_date, next_year, latest_close_price)

    # Check if the future data is not empty
    if x_test.shape[0] > 0:
        # load the weights from our best model
        model = tf.keras.models.load_model(os.path.join(inference_folder, 'model_weights.h5'))
        model.summary()

        # perform a prediction
        test_predictions_baseline = model.predict(x_test)
        test_predictions_baseline = min_max.inverse_transform(test_predictions_baseline)
        test_predictions_baseline = pd.DataFrame(test_predictions_baseline, columns=['Predicted_Price'])

        # Combine the predicted values with dates from the test data
        predicted_dates = pd.date_range(start=test_data.index[0], periods=len(test_predictions_baseline))
        test_predictions_baseline['Date'] = predicted_dates
        
        # Reset the index for proper concatenation
        test_data.reset_index(inplace=True)
        
        # Concatenate the test_data and predicted data
        combined_data = pd.concat([test_data, test_predictions_baseline], ignore_index=True)
        
        # Plotting predictions
        plt.figure(figsize=(14, 5))
        plt.plot(combined_data['Date'], combined_data.Close, color='green', label='Simulated [' + STOCK_TICKER + '] price')
        plt.plot(combined_data['Date'], combined_data['Predicted_Price'], color='red', label='Predicted [' + STOCK_TICKER + '] price')
        plt.xlabel('Time')
        plt.ylabel('Price [USD]')
        plt.legend()
        plt.title('Simulated vs Predicted Prices')
        plt.savefig(os.path.join(inference_folder, STOCK_TICKER + '_future_comparison.png'))
        plt.show()
    else:
        print("Error: Future data is empty.")

if __name__ == '__main__':
    TIME_STEPS = 3
    RUN_FOLDER = '^FTSE_20240103_edae6b8f5fc742031805151aeba98571'
    TOKEN = 'edae6b8f5fc742031805151aeba98571'
    STOCK_TICKER = '^FTSE'
    BATCH_SIZE = 10
    STOCK_START_DATE = pd.to_datetime('2017-11-01')
    start_date = pd.to_datetime('2017-01-01')
    end_date = datetime.today()
    duration = end_date - start_date
    STOCK_VALIDATION_DATE = start_date + 0.8 * duration
    GITHUB_URL = "https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning/raw/master/"
    EPOCHS = 100
    app.run(main)