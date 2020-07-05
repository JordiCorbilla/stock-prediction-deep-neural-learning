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

def main(argv):
    print(tf.version.VERSION)
    inference_folder = os.path.join(os.getcwd(), 'GOOG_20200704_b5f47746c83698528343678663ac3c96')

    # load future data
    





if __name__ == '__main__':
    app.run(main)