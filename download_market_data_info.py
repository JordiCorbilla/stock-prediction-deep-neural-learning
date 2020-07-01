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

import pandas as pd
import json
import yfinance as yf

msft = yf.Ticker("GOOG")

print('Info')
print(json.dumps(msft.info, indent=4, sort_keys=True))
print()
print('ISIN')
print(msft.isin)
print()
print('Major Holders')
print(msft.major_holders)
print()
print('Institutional Holders')
print(msft.institutional_holders)
print()
print('Dividents')
print(msft.dividends)
print()
print('Splits')
print(msft.splits)
print()
print('Actions')
print(msft.actions)
print()
print('Calendar')
print(msft.calendar)
print()
print('Recommendations')
print(msft.recommendations)
print()
print('Earnings')
print(msft.earnings)
print()
print('Quarterly Earnings')
print(msft.quarterly_earnings)
print()
print('Financials')
print(msft.financials)
print()
print('Quarterly Financials')
print(msft.quarterly_financials)
print()
print('Balance Sheet')
print(msft.balance_sheet)
print()
print('Quarterly Balance Sheet')
print(msft.quarterly_balance_sheet)
print()
print('Balancesheet')
print(msft.balancesheet)
print()
print('Quarterly Balancesheet')
print(msft.quarterly_balancesheet)
print()
print('Cashflow')
print(msft.cashflow)
print()
print('Quarterly Cashflow')
print(msft.quarterly_cashflow)
print()
print('Sustainability')
print(msft.sustainability)
print()
print('Options')
print(msft.options)