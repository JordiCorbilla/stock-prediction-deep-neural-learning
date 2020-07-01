# Stock prediction using deep neural learning

Predicting stock prices using a TensorFlow LSTM (long short-term memory) neural network for times series forecasting

## Introduction

Predicting stock prices is a cumbersome task as it does not follow any specific pattern. Changes in the stock prices are purely based on buy/sell actions during a period of time. In order to learn the specific characteristics of a stock price, we can use deep learning to identify these patterns through learning. One of the most well-known networks for series forecasting is [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) (long short-term memory) which is a recurrent neural network that is able to remember information over a long period of time, thus making them extremely useful for predicting stock prices.

## Stock Market Data

The initial data we will use for this model is taken directly from the [Yahoo Finance](https://finance.yahoo.com/quote/GOOG?p=GOOG) page which contains the latest market data on a specific stock price. To perform this operation easily using Python, we will use the [yFinance](https://aroussi.com/post/python-yahoo-finance) library which has been built specifically for this and that it will allow us to download all the information we need on a given [ticker symbol](https://www.investopedia.com/terms/t/tickersymbol.asp).

Below is a sample screenshot of the ticker symbol (GOOG) that we will use in this stock prediction article:
![](https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning/raw/master/samplestock.png)

### Market Info Download

To download the data info, we will need the yFinance library installed and then we will only need to perform the following operation to download all the relevant information of a given Stock using its ticker symbol.

Below is the output from the [download_market_data_info.py] file that is able to download financial data from Yahoo Finance. 

```cmd
C:\Users\thund\Source\Repos\stock-prediction-deep-neural-learning>python download_market_data_info.py
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

The data has a json document which we could use later on to create our Security Master if we ever wanted to store this data somewhere to keep track of the Securities we are going to trade with. As the data could come with different fields, my suggestion is to store them on a Data Lake so we can build it from multiple sources withouth having to worry to much on the way the data is structured.

### Market Data Download

The previous step helps us to identify several characteristiques of a given ticker symbol so we can use its properties to define some of the charts I'm showing below. Note that the yFinance library only requires the stock to download via ticker symbol, the start date and end date of the period we want to get. Additionally, we can also specify the granularity of the data using the interval parameter. By default the interval is 1 day and this is the one I will use for my training.

To download the data we can use the following command:

```python
start = pd.to_datetime('2004-08-01')
stock = ['GOOG']
data = yf.download(stock, start=start, end=datetime.date.today())
print(data)
```

And the sample output:

```cmd
C:\Users\thund\Source\Repos\stock-prediction-deep-neural-learning>python download_market_data.py
[*********************100%***********************]  1 of 1 completed
                   Open         High          Low        Close    Adj Close    Volume
Date
2004-08-19    49.813286    51.835709    47.800831    49.982655    49.982655  44871300
2004-08-20    50.316402    54.336334    50.062355    53.952770    53.952770  22942800
2004-08-23    55.168217    56.528118    54.321388    54.495735    54.495735  18342800
2004-08-24    55.412300    55.591629    51.591621    52.239193    52.239193  15319700
2004-08-25    52.284027    53.798351    51.746044    52.802086    52.802086   9232100
2004-08-26    52.279045    53.773445    52.134586    53.753517    53.753517   7128600
2004-08-27    53.848164    54.107193    52.647663    52.876804    52.876804   6241200
2004-08-30    52.443428    52.548038    50.814533    50.814533    50.814533   5221400
2004-08-31    50.958992    51.661362    50.889256    50.993862    50.993862   4941200
2004-09-01    51.158245    51.292744    49.648903    49.937820    49.937820   9181600
2004-09-02    49.409801    50.993862    49.285267    50.565468    50.565468  15190400
2004-09-03    50.286514    50.680038    49.474556    49.818268    49.818268   5176800
2004-09-07    50.316402    50.809555    49.619015    50.600338    50.600338   5875200
2004-09-08    50.181908    51.322632    50.062355    50.958992    50.958992   5009200
2004-09-09    51.073563    51.163227    50.311420    50.963974    50.963974   4080900
2004-09-10    50.610302    53.081039    50.460861    52.468334    52.468334   8740200
2004-09-13    53.115910    54.002586    53.031227    53.549286    53.549286   7881300
2004-09-14    53.524376    55.790882    53.195610    55.536835    55.536835  10880300
2004-09-15    55.073570    56.901718    54.894241    55.790882    55.790882  10763900
2004-09-16    55.960247    57.683788    55.616535    56.772205    56.772205   9310200
2004-09-17    56.996365    58.525631    56.562988    58.525631    58.525631   9517400
2004-09-20    58.256641    60.572956    58.166977    59.457142    59.457142  10679200
2004-09-21    59.681301    59.985161    58.535595    58.699978    58.699978   7263000
2004-09-22    58.480801    59.611561    58.186901    58.968971    58.968971   7617100
2004-09-23    59.198112    61.086033    58.291508    60.184414    60.184414   8576100
2004-09-24    60.244190    61.818291    59.656395    59.691261    59.691261   9166700
2004-09-27    59.556767    60.214302    58.680054    58.909195    58.909195   7099600
2004-09-28    60.423519    63.462128    59.880554    63.193138    63.193138  17009400
2004-09-29    63.113434    67.257904    62.879314    65.295258    65.295258  30661400
2004-09-30    64.707458    65.902977    64.259140    64.558022    64.558022  13823300
...                 ...          ...          ...          ...          ...       ...
2020-05-19  1386.996948  1392.000000  1373.484985  1373.484985  1373.484985   1280600
2020-05-20  1389.579956  1410.420044  1387.250000  1406.719971  1406.719971   1655400
2020-05-21  1408.000000  1415.489990  1393.449951  1402.800049  1402.800049   1385000
2020-05-22  1396.709961  1412.760010  1391.829956  1410.420044  1410.420044   1309400
2020-05-26  1437.270020  1441.000000  1412.130005  1417.020020  1417.020020   2060600
2020-05-27  1417.250000  1421.739990  1391.290039  1417.839966  1417.839966   1685800
2020-05-28  1396.859985  1440.839966  1396.000000  1416.729980  1416.729980   1692200
2020-05-29  1416.939941  1432.569946  1413.349976  1428.920044  1428.920044   1838100
2020-06-01  1418.390015  1437.959961  1418.000000  1431.819946  1431.819946   1217100
2020-06-02  1430.550049  1439.609985  1418.829956  1439.219971  1439.219971   1278100
2020-06-03  1438.300049  1446.552002  1429.776978  1436.380005  1436.380005   1256200
2020-06-04  1430.400024  1438.959961  1404.729980  1412.180054  1412.180054   1484300
2020-06-05  1413.170044  1445.050049  1406.000000  1438.390015  1438.390015   1734900
2020-06-08  1422.339966  1447.989990  1422.339966  1446.609985  1446.609985   1404200
2020-06-09  1445.359985  1468.000000  1443.209961  1456.160034  1456.160034   1409200
2020-06-10  1459.540039  1474.259033  1456.270020  1465.849976  1465.849976   1525200
2020-06-11  1442.479980  1454.474976  1402.000000  1403.839966  1403.839966   1991300
2020-06-12  1428.489990  1437.000000  1386.020020  1413.180054  1413.180054   1944200
2020-06-15  1390.800049  1424.800049  1387.920044  1419.849976  1419.849976   1503900
2020-06-16  1445.219971  1455.020020  1425.900024  1442.719971  1442.719971   1709200
2020-06-17  1447.160034  1460.000000  1431.380005  1451.119995  1451.119995   1548300
2020-06-18  1449.160034  1451.410034  1427.010010  1435.959961  1435.959961   1581900
2020-06-19  1444.000000  1447.800049  1421.349976  1431.719971  1431.719971   3157900
2020-06-22  1429.000000  1452.750000  1423.209961  1451.859985  1451.859985   1542400
2020-06-23  1455.640015  1475.941040  1445.239990  1464.410034  1464.410034   1429800
2020-06-24  1461.510010  1475.420044  1429.750000  1431.969971  1431.969971   1756000
2020-06-25  1429.900024  1442.900024  1420.000000  1441.329956  1441.329956   1230500
2020-06-26  1431.390015  1433.449951  1351.989990  1359.900024  1359.900024   4267700
2020-06-29  1358.180054  1395.599976  1347.010010  1394.969971  1394.969971   1810200
2020-06-30  1390.439941  1418.650024  1383.959961  1413.609985  1413.609985   2041600

[3994 rows x 6 columns]
```

Note that is important to mention the start date correctly just to ensure we are collecting data. If we don't do that we might end up having some NaN variables that could affect the output of our training.
