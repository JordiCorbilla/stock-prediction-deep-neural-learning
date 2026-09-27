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
import pandas as pd
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, blocking, project_folder, short_name, currency, stock_ticker):
        self.blocking = blocking
        self.project_folder = project_folder
        self.short_name = short_name
        self.currency = currency
        self.stock_ticker = stock_ticker

    def plot_histogram_data_split(self, training_data, test_data, validation_date):
        print("plotting Data and Histogram")
        plt.figure(figsize=(12, 5))
        plt.plot(training_data.Close, color='green')
        plt.plot(test_data.Close, color='red')
        plt.ylabel('Price [' + self.currency + ']')
        plt.xlabel("Date")
        plt.legend(["Pre-test data", "Held-out test data >= " + validation_date.strftime("%Y-%m-%d")])
        plt.title(self.short_name)
        plt.savefig(os.path.join(self.project_folder, self.short_name.strip().replace('.', '') + '_price.png'))

        fig, ax = plt.subplots()
        training_data.hist(ax=ax)
        fig.savefig(os.path.join(self.project_folder, self.short_name.strip().replace('.', '') + '_hist.png'))

        plt.pause(0.001)
        plt.show(block=self.blocking)

    def plot_loss(self, history):
        print("plotting loss")
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss/Validation Loss')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.project_folder, 'loss.png'))
        plt.pause(0.001)
        plt.show(block=self.blocking)

    def plot_mse(self, history):
        print("plotting MSE")
        metric_key = 'MSE' if 'MSE' in history.history else None
        if metric_key is None:
            candidates = [
                key for key in history.history
                if key.lower().endswith('mse') and not key.startswith('val_')
            ]
            metric_key = candidates[0] if candidates else None

        if metric_key is None:
            print("MSE metric is not present in this training history; skipping MSE plot.")
            return

        validation_key = 'val_' + metric_key
        plt.figure()
        plt.plot(history.history[metric_key], label=metric_key)
        if validation_key in history.history:
            plt.plot(history.history[validation_key], label=validation_key)
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('MSE/Validation MSE')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.project_folder, 'MSE.png'))
        plt.pause(0.001)
        plt.show(block=self.blocking)

    def plot_v9_architecture(self):
        """Render the v9 return-first architecture using the project's plotting layer."""
        print("plotting v9 architecture")
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.axis('off')

        boxes = [
            (0.04, 0.39, 0.20, 0.22, 'Scaled log-return\nwindow'),
            (0.31, 0.52, 0.16, 0.16, 'LSTM 128'),
            (0.31, 0.25, 0.16, 0.16, 'LSTM 64'),
            (0.57, 0.67, 0.18, 0.15, 'Direction head\nP(up)'),
            (0.57, 0.42, 0.18, 0.15, 'Return head\nE[r(t+1)]'),
            (0.57, 0.17, 0.18, 0.15, 'Quantile head\nq10 / q50 / q90'),
            (0.81, 0.39, 0.16, 0.22, 'Held-out\nevaluation'),
        ]

        for x0, y0, width, height, label in boxes:
            rectangle = plt.Rectangle((x0, y0), width, height, fill=False, linewidth=1.8)
            ax.add_patch(rectangle)
            ax.text(
                x0 + width / 2,
                y0 + height / 2,
                label,
                ha='center',
                va='center',
                fontsize=11,
            )

        def arrow(start, end):
            ax.annotate(
                '',
                xy=end,
                xytext=start,
                arrowprops={'arrowstyle': '->', 'lw': 1.5},
            )

        arrow((0.24, 0.50), (0.31, 0.60))
        arrow((0.39, 0.52), (0.39, 0.41))
        arrow((0.47, 0.33), (0.57, 0.745))
        arrow((0.47, 0.33), (0.57, 0.495))
        arrow((0.47, 0.33), (0.57, 0.245))
        arrow((0.75, 0.745), (0.81, 0.55))
        arrow((0.75, 0.495), (0.81, 0.50))
        arrow((0.75, 0.245), (0.81, 0.45))

        ax.set_title('v9 Return-First Multi-Task Architecture', fontsize=15, pad=15)
        fig.savefig(os.path.join(self.project_folder, 'v9_architecture.png'), bbox_inches='tight')
        plt.pause(0.001)
        plt.show(block=self.blocking)

    def project_plot_predictions(self, price_predicted, test_data):
        print("plotting predictions")
        predicted_series = pd.to_numeric(price_predicted[self.stock_ticker + '_predicted'], errors='coerce')
        actual_values = test_data['Close']
        if isinstance(actual_values, pd.DataFrame):
            actual_values = actual_values.iloc[:, 0]
        actual_series = pd.to_numeric(actual_values, errors='coerce')
        plt.figure(figsize=(14, 5))
        plt.plot(predicted_series.to_numpy(), color='red', label='Predicted [' + self.short_name + '] price')
        plt.plot(actual_series.to_numpy(), color='green', label='Actual [' + self.short_name + '] price')
        plt.xlabel('Time')
        plt.ylabel('Price [' + self.currency + ']')
        plt.legend()
        plt.title('Prediction')
        plt.savefig(os.path.join(self.project_folder, self.short_name.strip().replace('.', '') + '_prediction.png'))
        plt.pause(0.001)
        plt.show(block=self.blocking)
