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

import json
import yfinance as yf

sec = yf.Ticker("^FTSE")

data = sec.history()
#data.head()

my_max = data['Close'].idxmax()
my_min = data['Close'].idxmin()

print('Info')
print(json.dumps(sec.info, indent=4, sort_keys=True))
print()
print('ISIN')
print(sec.isin)
print()
print('Major Holders')
print(sec.major_holders)
print()
print('Institutional Holders')
print(sec.institutional_holders)
print()
print('Dividents')
print(sec.dividends)
print()
print('Splits')
print(sec.splits)
print()
print('Actions')
print(sec.actions)
print()
print('Calendar')
print(sec.calendar)
print()
print('Recommendations')
print(sec.recommendations)
print()
print('Earnings')
print(sec.earnings)
print()
print('Quarterly Earnings')
print(sec.quarterly_earnings)
print()
print('Financials')
print(sec.financials)
print()
print('Quarterly Financials')
print(sec.quarterly_financials)
print()
print('Balance Sheet')
print(sec.balance_sheet)
print()
print('Quarterly Balance Sheet')
print(sec.quarterly_balance_sheet)
print()
print('BalanceSheet')
print(sec.balancesheet)
print()
print('Quarterly BalanceSheet')
print(sec.quarterly_balancesheet)
print()
print('Cash Flow')
print(sec.cashflow)
print()
print('Quarterly Cash Flow')
print(sec.quarterly_cashflow)
print()
print('Sustainability')
print(sec.sustainability)
print()
print('Options')
print(sec.options)