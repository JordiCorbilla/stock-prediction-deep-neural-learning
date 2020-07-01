# Stock prediction using deep neural learning

Predicting stock prices using a TensorFlow LSTM (long short-term memory) neural network for times series forecasting

## Introduction

Predicting stock prices is a cumbersome task as it does not follow any specific pattern. Changes in the stock prices are purely based on buy/sell actions during a period of time. In order to learn the specific characteristics of a stock price, we can use deep learning to identify these patterns through learning. One of the most well-known networks for series forecasting is [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) (long short-term memory) which is a recurrent neural network that is able to remember information over a long period of time, thus making them extremely useful for predicting stock prices.

## Stock Market Data

The initial data we will use for this model is taken directly from the [Yahoo Finance](https://finance.yahoo.com/quote/GOOG?p=GOOG) page which contains the latest market data on a specific stock price. To perform this operation easily using Python, we will use the [yFinance](https://aroussi.com/post/python-yahoo-finance) library which has been built specifically for this and that it will allow us to download all the information we need on a given [ticker symbol](https://www.investopedia.com/terms/t/tickersymbol.asp).

Below is a sample screenshot of the ticker symbol (GOOG) that we will use in this stock prediction article:
![](https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning/raw/master/samplestock.png)

### Market Data Download

To download the data, we will need the yFinance library installed and then we will only need to perform the following operation to download all the relevant information of a given Stock using its ticker symbol.

Below is the output from the [download_market_data.py] file that is able to download financial data from Yahoo Finance. 

```cmd
C:\Users\thund\Source\Repos\stock-prediction-deep-neural-learning>python download_market_data.py
Info
{
    "52WeekChange": 0.26037383,
    "SandP52WeekChange": 0.034871936,
    "address1": "1600 Amphitheatre Parkway",
    "algorithm": null,
    "annualHoldingsTurnover": null,
    "annualReportExpenseRatio": null,
    "ask": 1432.77,
    "askSize": 1400,
    "averageDailyVolume10Day": 2011171,
    "averageVolume": 1857809,
    "averageVolume10days": 2011171,
    "beta": 1.068946,
    "beta3Year": null,
    "bid": 1432.16,
    "bidSize": 3000,
    "bookValue": 297.759,
    "category": null,
    "circulatingSupply": null,
    "city": "Mountain View",
    "companyOfficers": [],
    "country": "United States",
    "currency": "USD",
    "dateShortInterest": 1592179200,
    "dayHigh": 1441.19,
    "dayLow": 1409.82,
    "dividendRate": null,
    "dividendYield": null,
    "earningsQuarterlyGrowth": 0.027,
    "enterpriseToEbitda": 17.899,
    "enterpriseToRevenue": 5.187,
    "enterpriseValue": 864533741568,
    "exDividendDate": null,
    "exchange": "NMS",
    "exchangeTimezoneName": "America/New_York",
    "exchangeTimezoneShortName": "EDT",
    "expireDate": null,
    "fiftyDayAverage": 1417.009,
    "fiftyTwoWeekHigh": 1532.106,
    "fiftyTwoWeekLow": 1013.536,
    "fiveYearAverageReturn": null,
    "fiveYearAvgDividendYield": null,
    "floatShares": 613293304,
    "forwardEps": 55.05,
    "forwardPE": 26.028149,
    "fromCurrency": null,
    "fullTimeEmployees": 123048,
    "fundFamily": null,
    "fundInceptionDate": null,
    "gmtOffSetMilliseconds": "-14400000",
    "heldPercentInsiders": 0.05746,
    "heldPercentInstitutions": 0.7062,
    "industry": "Internet Content & Information",
    "isEsgPopulated": false,
    "lastCapGain": null,
    "lastDividendValue": null,
    "lastFiscalYearEnd": 1577750400,
    "lastMarket": null,
    "lastSplitDate": 1430092800,
    "lastSplitFactor": "10000000:10000000",
    "legalType": null,
    "logo_url": "https://logo.clearbit.com/abc.xyz",
    "longBusinessSummary": "Alphabet Inc. provides online advertising services in the United States, Europe, the Middle East, Africa, the Asia-Pacific, Canada, and Latin America. It offers performance and brand advertising services. The company operates through Google and Other Bets segments. The Google segment offers products, such as Ads, Android, Chrome, Google Cloud, Google Maps, Google Play, Hardware, Search, and YouTube, as well as technical infrastructure. It also offers digital content, cloud services, hardware devices, and other miscellaneous products and services. The Other Bets segment includes businesses, including Access, Calico, CapitalG, GV, Verily, Waymo, and X, as well as Internet and television services. Alphabet Inc. was founded in 1998 and is headquartered in Mountain View, California.",
    "longName": "Alphabet Inc.",
    "market": "us_market",
    "marketCap": 979650805760,
    "maxAge": 1,
    "maxSupply": null,
    "messageBoardId": "finmb_29096",
    "morningStarOverallRating": null,
    "morningStarRiskRating": null,
    "mostRecentQuarter": 1585612800,
    "navPrice": null,
    "netIncomeToCommon": 34522001408,
    "nextFiscalYearEnd": 1640908800,
    "open": 1411.1,
    "openInterest": null,
    "payoutRatio": 0,
    "pegRatio": 4.38,
    "phone": "650-253-0000",
    "previousClose": 1413.61,
    "priceHint": 2,
    "priceToBook": 4.812112,
    "priceToSalesTrailing12Months": 5.87754,
    "profitMargins": 0.20712,
    "quoteType": "EQUITY",
    "regularMarketDayHigh": 1441.19,
    "regularMarketDayLow": 1409.82,
    "regularMarketOpen": 1411.1,
    "regularMarketPreviousClose": 1413.61,
    "regularMarketPrice": 1411.1,
    "regularMarketVolume": 1084440,
    "revenueQuarterlyGrowth": null,
    "sector": "Communication Services",
    "sharesOutstanding": 336161984,
    "sharesPercentSharesOut": 0.0049,
    "sharesShort": 3371476,
    "sharesShortPreviousMonthDate": 1589500800,
    "sharesShortPriorMonth": 3462105,
    "shortName": "Alphabet Inc.",
    "shortPercentOfFloat": null,
    "shortRatio": 1.9,
    "startDate": null,
    "state": "CA",
    "strikePrice": null,
    "symbol": "GOOG",
    "threeYearAverageReturn": null,
    "toCurrency": null,
    "totalAssets": null,
    "tradeable": false,
    "trailingAnnualDividendRate": null,
    "trailingAnnualDividendYield": null,
    "trailingEps": 49.572,
    "trailingPE": 28.904415,
    "twoHundredDayAverage": 1352.9939,
    "volume": 1084440,
    "volume24Hr": null,
    "volumeAllCurrencies": null,
    "website": "http://www.abc.xyz",
    "yield": null,
    "ytdReturn": null,
    "zip": "94043"
}

ISIN
-

Major Holders
        0                                      1
0   5.75%        % of Shares Held by All Insider
1  70.62%       % of Shares Held by Institutions
2  74.93%        % of Float Held by Institutions
3    3304  Number of Institutions Holding Shares

Institutional Holders
                            Holder    Shares Date Reported   % Out        Value
0       Vanguard Group, Inc. (The)  23162950    2020-03-30  0.0687  26934109889
1                   Blackrock Inc.  20264225    2020-03-30  0.0601  23563443472
2    Price (T.Rowe) Associates Inc  12520058    2020-03-30  0.0371  14558448642
3         State Street Corporation  11814026    2020-03-30  0.0350  13737467573
4                         FMR, LLC   8331868    2020-03-30  0.0247   9688379429
5  Capital International Investors   4555880    2020-03-30  0.0135   5297622822
6    Geode Capital Management, LLC   4403934    2020-03-30  0.0131   5120938494
7       Northern Trust Corporation   4017009    2020-03-30  0.0119   4671018235
8        JP Morgan Chase & Company   3707376    2020-03-30  0.0110   4310973886
9          AllianceBernstein, L.P.   3483382    2020-03-30  0.0103   4050511423

Dividents
Series([], Name: Dividends, dtype: int64)

Splits
Date
2014-03-27    2.002
2015-04-27    1.000
Name: Stock Splits, dtype: float64

Actions
            Dividends  Stock Splits
Date
2014-03-27        0.0         2.002
2015-04-27        0.0         1.000

Calendar
Empty DataFrame
Columns: []
Index: [Earnings Date, Earnings Average, Earnings Low, Earnings High, Revenue Average, Revenue Low, Revenue High]

Recommendations
                                         Firm    To Grade    From Grade Action
Date
2012-03-14 15:28:00                Oxen Group        Hold                 init
2012-03-28 06:29:00                 Citigroup         Buy                 main
2012-04-03 08:45:00  Global Equities Research  Overweight                 main
2012-04-05 06:34:00             Deutsche Bank         Buy                 main
2012-04-09 06:03:00          Pivotal Research         Buy                 main
2012-04-10 11:32:00                       UBS         Buy                 main
2012-04-13 06:16:00             Deutsche Bank         Buy                 main
2012-04-13 06:18:00                 Jefferies         Buy                 main
2012-04-13 06:37:00              PiperJaffray  Overweight                 main
2012-04-13 06:38:00             Goldman Sachs     Neutral                 main
2012-04-13 06:41:00                 JP Morgan  Overweight                 main
2012-04-13 06:51:00               Oppenheimer  Outperform                 main
2012-04-13 07:13:00                 Benchmark        Hold                 main
2012-04-13 08:46:00               BMO Capital  Outperform                 main
2012-04-16 06:52:00            Hilliard Lyons         Buy                 main
2012-06-06 06:17:00             Deutsche Bank         Buy                 main
2012-06-06 06:56:00                 JP Morgan  Overweight                 main
2012-06-22 06:15:00                 Citigroup         Buy                 main
2012-07-13 05:57:00                   Wedbush     Neutral                 init
2012-07-17 09:33:00                            Outperform                 main
2012-07-20 06:43:00                 Benchmark        Hold                 main
2012-07-20 06:54:00             Deutsche Bank         Buy                 main
2012-07-20 06:59:00           Bank of America         Buy                 main
2012-08-13 05:49:00            Morgan Stanley  Overweight  Equal-Weight     up
2012-09-17 06:07:00  Global Equities Research  Overweight                 main
2012-09-21 06:28:00         Cantor Fitzgerald         Buy                 init
2012-09-24 06:11:00                 Citigroup         Buy                 main
2012-09-24 09:05:00          Pivotal Research         Buy                 main
2012-09-25 07:20:00                  Capstone         Buy                 main
2012-09-26 05:48:00         Canaccord Genuity         Buy                 main
...                                       ...         ...           ...    ...
2017-10-27 19:29:31                       UBS         Buy                 main
2018-02-02 14:04:52              PiperJaffray  Overweight    Overweight   main
2018-04-24 11:43:49                 JP Morgan  Overweight    Overweight   main
2018-04-24 12:24:37             Deutsche Bank         Buy           Buy   main
2018-05-05 14:00:37              B. Riley FBR         Buy                 main
2018-07-13 13:49:13               Cowen & Co.  Outperform    Outperform   main
2018-07-24 11:50:55               Cowen & Co.  Outperform    Outperform   main
2018-07-24 13:33:47             Raymond James  Outperform    Outperform   main
2018-10-23 11:18:00             Deutsche Bank         Buy           Buy   main
2018-10-26 15:17:08             Raymond James  Outperform    Outperform   main
2019-01-23 12:55:04             Deutsche Bank         Buy           Buy   main
2019-02-05 12:55:12             Deutsche Bank         Buy           Buy   main
2019-02-05 13:18:47              PiperJaffray  Overweight    Overweight   main
2019-05-15 12:34:54             Deutsche Bank         Buy                 main
2019-10-23 12:58:59             Credit Suisse  Outperform                 main
2019-10-29 11:58:09             Raymond James  Outperform                 main
2019-10-29 14:15:40             Deutsche Bank         Buy                 main
2019-10-29 15:48:29                       UBS         Buy                 main
2020-01-06 11:22:07          Pivotal Research         Buy          Hold     up
2020-01-17 13:01:48                       UBS         Buy                 main
2020-02-04 12:26:56             Piper Sandler  Overweight                 main
2020-02-04 12:41:00             Raymond James  Outperform                 main
2020-02-04 14:00:36             Deutsche Bank         Buy                 main
2020-02-06 11:34:20                      CFRA  Strong Buy                 main
2020-03-18 13:52:51                 JP Morgan  Overweight                 main
2020-03-30 13:26:16                       UBS         Buy                 main
2020-04-17 13:01:41               Oppenheimer  Outperform                 main
2020-04-20 19:29:50             Credit Suisse  Outperform                 main
2020-04-29 14:01:51                       UBS         Buy                 main
2020-05-05 12:44:16             Deutsche Bank         Buy                 main

[219 rows x 4 columns]

Earnings
Empty DataFrame
Columns: [Open, High, Low, Close, Adj Close, Volume]
Index: []

Quarterly Earnings
Empty DataFrame
Columns: [Open, High, Low, Close, Adj Close, Volume]
Index: []

Financials
Empty DataFrame
Columns: [Open, High, Low, Close, Adj Close, Volume]
Index: []

Quarterly Financials
Empty DataFrame
Columns: [Open, High, Low, Close, Adj Close, Volume]
Index: []

Balance Sheet
Empty DataFrame
Columns: [Open, High, Low, Close, Adj Close, Volume]
Index: []

Quarterly Balance Sheet
Empty DataFrame
Columns: [Open, High, Low, Close, Adj Close, Volume]
Index: []

Balancesheet
Empty DataFrame
Columns: [Open, High, Low, Close, Adj Close, Volume]
Index: []

Quarterly Balancesheet
Empty DataFrame
Columns: [Open, High, Low, Close, Adj Close, Volume]
Index: []

Cashflow
Empty DataFrame
Columns: [Open, High, Low, Close, Adj Close, Volume]
Index: []

Quarterly Cashflow
Empty DataFrame
Columns: [Open, High, Low, Close, Adj Close, Volume]
Index: []

Sustainability
None

Options
('2020-07-02', '2020-07-10', '2020-07-17', '2020-07-24', '2020-07-31', '2020-08-07', '2020-08-21', '2020-09-18', '2020-11-20', '2020-12-01', '2020-12-18', '2021-01-15', '2021-06-18', '2022-01-21', '2022-06-17')
```
