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


class StockPrediction:
    def __init__(self, ticker, start_date, validation_date, project_folder, github_url, epochs, time_steps, token, batch_size):
        self._ticker = ticker
        self._start_date = start_date
        self._validation_date = validation_date
        self._project_folder = project_folder
        self._github_url = github_url
        self._epochs = epochs
        self._time_steps = time_steps
        self._token = token
        self._batch_size = batch_size

    def get_ticker(self):
        return self._ticker

    def set_ticker(self, value):
        self._ticker = value

    def get_start_date(self):
        return self._start_date

    def set_start_date(self, value):
        self._start_date = value

    def get_validation_date(self):
        return self._validation_date

    def set_validation_date(self, value):
        self._validation_date = value

    def get_project_folder(self):
        return self._project_folder

    def set_project_folder(self, value):
        self._project_folder = value
        
    def set_github_url(self, value):
        self._github_url = value
        
    def get_github_url(self):
        return self._github_url
    
    def get_epochs(self):
        return self._epochs

    def get_time_steps(self):
        return self._time_steps
        
    def get_token(self):
        return self._token     
    
    def get_batch_size(self):
        return self._batch_size     