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


class ReadmeGenerator:
    def __init__(self, project_folder, short_name):
        self.baseUrl = "https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning/raw/master/"
        self.project_folder = project_folder
        self.short_name = short_name.strip().replace('.', '').replace(' ', '%20')

    def write(self):
        my_file = open(os.path.join(self.project_folder, 'README.md'), "w+")
        my_file.write('![](' + self.baseUrl + self.project_folder + '/' + self.short_name + '_price.png)\n')
        my_file.write('![](' + self.baseUrl + self.project_folder + '/' + self.short_name + '_hist.png)\n')
        my_file.write('![](' + self.baseUrl + self.project_folder + '/' + self.short_name + '_prediction.png)\n')
        my_file.write('![](' + self.baseUrl + self.project_folder + '/' + 'MSE.png)\n')
        my_file.write('![](' + self.baseUrl + self.project_folder + '/' + 'loss.png)\n')
