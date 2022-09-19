# Stock prediction using deep neural learning

Predicting stock prices using a TensorFlow LSTM (long short-term memory) neural network for times series forecasting.

## 1) Introduction

Predicting stock prices is a cumbersome task as it does not follow any specific pattern. Changes in the stock prices are purely based on supply and demand during a period of time. In order to learn the specific characteristics of a stock price, we can use deep learning to identify these patterns through machine learning. One of the most well-known networks for series forecasting is [LSTM](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory) (long short-term memory) which is a Recurrent Neural Network (RNN) that is able to remember information over a long period of time, thus making them extremely useful for predicting stock prices. RNNs are well-suited to time series data and they are able to process the data step-by-step, maintaining an internal state where they cache the information they have seen so far in a summarised version. The successful prediction of a stock's future price could yield a significant profit.

## 2) Stock Market Data

The initial data we will use for this model is taken directly from the [Yahoo Finance](https://finance.yahoo.com/quote/GOOG?p=GOOG) page which contains the latest market data on a specific stock price. To perform this operation easily using Python, we will use the [yFinance](https://aroussi.com/post/python-yahoo-finance) library which has been built specifically for this and that it will allow us to download all the information we need on a given [ticker symbol](https://www.investopedia.com/terms/t/tickersymbol.asp).

Below is a sample screenshot of the ticker symbol (GOOG) that we will use in this stock prediction article:
![](https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning/raw/master/samplestock.png)

### 2.1) Market Info Download

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

The data has a JSON document that we could use later on to create our Security Master if we ever wanted to store this data somewhere to keep track of the Securities we are going to trade with. As the data could come with different fields, my suggestion is to store them on a Data Lake so we can build it from multiple sources without having to worry too much about the way the data is structured.

### 2.2) Market Data Download

The previous step helps us to identify several characteristics of a given ticker symbol so we can use its properties to define some of the charts I'm showing below. Note that the yFinance library only requires the stock to download via ticker symbol, the start date and end date of the period we want to get. Additionally, we can also specify the granularity of the data using the interval parameter. By default, the interval is 1 day and this is the one I will use for my training.

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

## 3) Deep Learning Model

### 3.1) Training and Validation Data

Now that we have the data that we want to use, we need to define what defines our traning and validation data. As stocks could vary depending on the dates, the function I have created requires 3 basic arguments:
- Ticker Symbol: **GOOG**
- Start Date: Date as to when they started, in this case, it was **2004-Aug-01**.
- Validation Date: Date as to when we want the validation to be considered. In this case, we specify **2017-01-01** as our data point.

Note that you will need to have configured [TensorFlow](https://www.tensorflow.org/), Keras ,and a GPU in order to run the samples below.

In this exercise, I'm only interested in the [closing price](https://www.investopedia.com/terms/c/closingprice.asp) which is the standard benchmark regarding stocks or securities.

Below you can find the chart with the division we will create between Training Data and Validation Data:
![](https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning/raw/master/TrainingDataValidationData.png)

Also, the histogram showing the distribution of the prices:

![](https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning/raw/master/DataHistogram.png)

### 3.2) Data Normalization

In order to normalise the data, we need to scale it between 0 and 1 so we talk on a common scale. To accomplish this, we can use the preprocessing tool [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) as seen below:

```python
    min_max = MinMaxScaler(feature_range=(0, 1))
    train_scaled = min_max.fit_transform(training_data)
```

### 3.3) Adding Timesteps

LSTM network needs the data imported as a 3D array. To translate this 2D array into a 3D one, we use a short [timestep](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) to loop through the data and create smaller partitions and feed them into the model. The final array is then reshaped into training samples, x number of timesteps, and 1 feature per step. The code below represents this concept:

```python
    time_steps = 3
    for i in range(time_steps, train_scaled.shape[0]):
        x_train.append(train_scaled[i - time_steps:i])
        y_train.append(train_scaled[i, 0])
```

![](https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning/blob/master/3dmatrix.png)

We have implemented a time step of 3 days. Using this technique, we allow our network to look back 3 days on our data to predict the subsequent day). The figure below represents how our implementation uses this concept and how the first 3 samples for Close price would generate the 4th sample and so on.This will generate a matrix of shape (3,1), 3 being the time steps and 1 the number of features (Close price).

![](https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning/blob/master/timesteps.png)

### 3.4) Creation of the deep learning model LSTM

To create this model, you will need to have **TensorFlow**, **TensorFlow-Gpu** and **Keras** installed in order for this to run. The code for this model can be seen below and the explanation for each layer is also defined below:

```python
def create_long_short_term_memory_model(x_train):
    model = Sequential()
    # 1st layer with Dropout regularisation
    # * units = add 100 neurons is the dimensionality of the output space
    # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
    # * input_shape => Shape of the training dataset
    model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    # 20% of the layers will be dropped
    model.add(Dropout(0.2))
    # 2nd LSTM layer
    # * units = add 50 neurons is the dimensionality of the output space
    # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
    model.add(LSTM(units=50, return_sequences=True))
    # 20% of the layers will be dropped
    model.add(Dropout(0.2))
    # 3rd LSTM layer
    # * units = add 50 neurons is the dimensionality of the output space
    # * return_sequences = True to stack LSTM layers so the next LSTM layer has a three-dimensional sequence input
    model.add(LSTM(units=50, return_sequences=True))
    # 50% of the layers will be dropped
    model.add(Dropout(0.5))
    # 4th LSTM layer
    # * units = add 50 neurons is the dimensionality of the output space
    model.add(LSTM(units=50))
    # 50% of the layers will be dropped
    model.add(Dropout(0.5))
    # Dense layer that specifies an output of one unit
    model.add(Dense(units=1))
    model.summary()
    tf.keras.utils.plot_model(model, to_file=os.path.join(project_folder, 'model_lstm.png'), show_shapes=True,
                              show_layer_names=True)
    return model
```

The rendered model can be seen in the image below, producing a model with more than 100k trainable parameters.
![](https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning/raw/master/model_lstm.png)

```cmd
Layer (type)                 Output Shape              Param #
=================================================================
lstm_1 (LSTM)                (None, 60, 100)           40800
_________________________________________________________________
dropout_1 (Dropout)          (None, 60, 100)           0
_________________________________________________________________
lstm_2 (LSTM)                (None, 60, 50)            30200
_________________________________________________________________
dropout_2 (Dropout)          (None, 60, 50)            0
_________________________________________________________________
lstm_3 (LSTM)                (None, 60, 50)            20200
_________________________________________________________________
dropout_3 (Dropout)          (None, 60, 50)            0
_________________________________________________________________
lstm_4 (LSTM)                (None, 50)                20200
_________________________________________________________________
dropout_4 (Dropout)          (None, 50)                0
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 51
=================================================================
Total params: 111,451
Trainable params: 111,451
Non-trainable params: 0
```

Once we have defined the model, we need to specify the metrics we want to use to track how well our model is behaving and also the kind of optimizer we want to use for our training. I have also defined the patience I want my model to have and what is the rule defined for it.

```python
    defined_metrics = [
        tf.keras.metrics.MeanSquaredError(name='MSE')
    ]

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=defined_metrics)
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test),
                        callbacks=[callback])
```

This model is slightly fined tuned to reach the lowest validation loss. In this example, we reach a validation loss of 0.14% with an MSE (Mean Square Error) of  0.14% which is relatively good, providing us with a very accurate result.

The training result can be seen below:

```cmd
Train on 3055 samples, validate on 881 samples
Epoch 1/100
2020-07-11 15:15:34.557035: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cublas64_100.dll
3112/3112 [==============================] - 19s 6ms/sample - loss: 0.0451 - MSE: 0.0451 - val_loss: 0.0068 - val_MSE: 0.0068
Epoch 2/100
3112/3112 [==============================] - 4s 1ms/sample - loss: 0.0088 - MSE: 0.0088 - val_loss: 0.0045 - val_MSE: 0.0045
Epoch 3/100
3112/3112 [==============================] - 5s 1ms/sample - loss: 0.0062 - MSE: 0.0062 - val_loss: 0.0032 - val_MSE: 0.0032
Epoch 4/100
3112/3112 [==============================] - 5s 1ms/sample - loss: 0.0051 - MSE: 0.0051 - val_loss: 0.0015 - val_MSE: 0.0015
Epoch 5/100
3112/3112 [==============================] - 7s 2ms/sample - loss: 0.0045 - MSE: 0.0045 - val_loss: 0.0013 - val_MSE: 0.0013
Epoch 6/100
3112/3112 [==============================] - 5s 2ms/sample - loss: 0.0045 - MSE: 0.0045 - val_loss: 0.0013 - val_MSE: 0.0013
Epoch 7/100
3112/3112 [==============================] - 5s 2ms/sample - loss: 0.0045 - MSE: 0.0045 - val_loss: 0.0015 - val_MSE: 0.0015
Epoch 8/100
3112/3112 [==============================] - 5s 1ms/sample - loss: 0.0040 - MSE: 0.0040 - val_loss: 0.0015 - val_MSE: 0.0015
Epoch 9/100
3112/3112 [==============================] - 5s 1ms/sample - loss: 0.0039 - MSE: 0.0039 - val_loss: 0.0014 - val_MSE: 0.0014
Epoch 00009: early stopping
saving weights
plotting loss
plotting MSE
display the content of the model
886/1 - 0s - loss: 0.0029 - MSE: 0.0014
loss :  0.0014113364930413916
MSE :  0.0014113366
```

### 3.5) Making predictions happen

Now it is time to prepare our testing data and send it through our deep-learning model to obtain the predictions we are trying to get.

First, we need to import the test data using the same approach we used for the training data using the time steps:

```python
    # Testing Data Transformation
    x_test = []
    y_test = []
    for i in range(time_steps, test_scaled.shape[0]):
        x_test.append(test_scaled[i - time_steps:i])
        y_test.append(test_scaled[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
```

Now we can call the predict method which will allow us to generate the stock prediction based on the training done over the training data. As a result, we will generate a csv file that contains the result of the prediction and also a chart that shows what's the real vs the estimation. 

![](https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning/raw/master/Alphabet%20Inc_prediction.png)

With the validation loss and validation MSE metrics:

![](https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning/raw/master/loss.png)
![](https://github.com/JordiCorbilla/stock-prediction-deep-neural-learning/raw/master/MSE.png)

# 4) Usage

This has been built using Python 3.8 version.

Download the source code and install the following packages:

```cmd
C:\Users\thund\Source\Repos\stock-prediction-deep-neural-learning>pip list
Package              Version
-------------------- ---------
absl-py                            0.13.0
alabaster                          0.7.12
anaconda-client                    1.7.2
anaconda-navigator                 2.0.3
anaconda-project                   0.9.1
anyio                              2.2.0
appdirs                            1.4.4
argh                               0.26.2
argon2-cffi                        20.1.0
asn1crypto                         1.4.0
astroid                            2.6.6
astropy                            4.2.1
astunparse                         1.6.3
async-generator                    1.10
atomicwrites                       1.4.0
attrs                              20.3.0
autopep8                           1.5.5
Babel                              2.9.0
backcall                           0.2.0
backports.functools-lru-cache      1.6.4
backports.shutil-get-terminal-size 1.0.0
backports.tempfile                 1.0
backports.weakref                  1.0.post1
bcrypt                             3.2.0
beautifulsoup4                     4.9.3
bitarray                           1.9.2
bkcharts                           0.2
black                              19.10b0
bleach                             3.3.0
bokeh                              2.3.2
boto                               2.49.0
Bottleneck                         1.3.2
brotlipy                           0.7.0
cachetools                         4.2.2
certifi                            2020.12.5
cffi                               1.14.5
chardet                            4.0.0
clang                              5.0
click                              7.1.2
cloudpickle                        1.6.0
clyent                             1.2.2
colorama                           0.4.4
comtypes                           1.1.9
conda                              4.10.1
conda-build                        3.21.4
conda-content-trust                0+unknown
conda-package-handling             1.7.3
conda-repo-cli                     1.0.4
conda-token                        0.3.0
conda-verify                       3.4.2
contextlib2                        0.6.0.post1
cryptography                       3.4.7
cycler                             0.10.0
Cython                             0.29.23
cytoolz                            0.11.0
dask                               2021.4.0
decorator                          5.0.6
defusedxml                         0.7.1
diff-match-patch                   20200713
distributed                        2021.4.0
docutils                           0.17
entrypoints                        0.3
et-xmlfile                         1.0.1
fastcache                          1.1.0
filelock                           3.0.12
flake8                             3.8.4
Flask                              1.1.2
flatbuffers                        1.12
fsspec                             0.9.0
future                             0.18.2
gast                               0.4.0
gevent                             21.1.2
glob2                              0.7
google-auth                        1.35.0
google-auth-oauthlib               0.4.6
google-pasta                       0.2.0
greenlet                           1.0.0
grpcio                             1.40.0
h5py                               3.1.0
HeapDict                           1.0.1
helpdev                            0.7.1
html5lib                           1.1
idna                               2.10
imagecodecs                        2021.3.31
imageio                            2.9.0
imagesize                          1.2.0
importlib-metadata                 3.10.0
iniconfig                          1.1.1
intervaltree                       3.1.0
ipykernel                          5.3.4
ipython                            7.22.0
ipython-genutils                   0.2.0
ipywidgets                         7.6.3
isort                              5.8.0
itsdangerous                       1.1.0
jdcal                              1.4.1
jedi                               0.17.2
Jinja2                             2.11.3
joblib                             1.0.1
json5                              0.9.5
jsonschema                         3.2.0
jupyter                            1.0.0
jupyter-client                     6.1.12
jupyter-console                    6.4.0
jupyter-core                       4.7.1
jupyter-packaging                  0.7.12
jupyter-server                     1.4.1
jupyterlab                         3.0.14
jupyterlab-pygments                0.1.2
jupyterlab-server                  2.4.0
jupyterlab-widgets                 1.0.0
keras                              2.6.0
Keras-Preprocessing                1.1.2
keyring                            22.3.0
kiwisolver                         1.3.1
lazy-object-proxy                  1.6.0
libarchive-c                       2.9
llvmlite                           0.36.0
locket                             0.2.1
lxml                               4.6.3
Markdown                           3.3.4
MarkupSafe                         1.1.1
matplotlib                         3.3.4
mccabe                             0.6.1
menuinst                           1.4.16
mistune                            0.8.4
mkl-fft                            1.3.0
mkl-random                         1.2.1
mkl-service                        2.3.0
mock                               4.0.3
more-itertools                     8.7.0
mpmath                             1.2.1
msgpack                            1.0.2
multipledispatch                   0.6.0
multitasking                       0.0.9
mypy-extensions                    0.4.3
navigator-updater                  0.2.1
nbclassic                          0.2.6
nbclient                           0.5.3
nbconvert                          6.0.7
nbformat                           5.1.3
nest-asyncio                       1.5.1
networkx                           2.5
nltk                               3.6.1
nose                               1.3.7
notebook                           6.3.0
numba                              0.53.1
numexpr                            2.7.3
numpy                              1.21.2
numpydoc                           1.1.0
oauthlib                           3.1.1
olefile                            0.46
openpyxl                           3.0.7
opt-einsum                         3.3.0
packaging                          20.9
pandas                             1.2.4
pandocfilters                      1.4.3
paramiko                           2.7.2
parso                              0.7.0
partd                              1.2.0
path                               15.1.2
pathlib2                           2.3.5
pathspec                           0.7.0
patsy                              0.5.1
pep8                               1.7.1
pexpect                            4.8.0
pickleshare                        0.7.5
Pillow                             8.2.0
pip                                21.2.4
pkginfo                            1.7.0
pluggy                             0.13.1
ply                                3.11
prometheus-client                  0.10.1
prompt-toolkit                     3.0.17
protobuf                           3.18.0
psutil                             5.8.0
ptyprocess                         0.7.0
py                                 1.10.0
pyasn1                             0.4.8
pyasn1-modules                     0.2.8
pycodestyle                        2.6.0
pycosat                            0.6.3
pycparser                          2.20
pycurl                             7.43.0.6
pydocstyle                         6.0.0
pyerfa                             1.7.3
pyflakes                           2.2.0
Pygments                           2.8.1
pylint                             2.7.4
pyls-black                         0.4.6
pyls-spyder                        0.3.2
PyNaCl                             1.4.0
pyodbc                             4.0.0-unsupported
pyOpenSSL                          20.0.1
pyparsing                          2.4.7
PyQt5                              5.12.3
PyQt5-sip                          12.9.0
PyQtWebEngine                      5.12.1
pyreadline                         2.1
pyrsistent                         0.17.3
PySocks                            1.7.1
pytest                             6.2.3
python-dateutil                    2.8.1
python-jsonrpc-server              0.4.0
python-language-server             0.36.2
pytz                               2021.1
PyWavelets                         1.1.1
pywin32                            227
pywin32-ctypes                     0.2.0
pywinpty                           0.5.7
PyYAML                             5.4.1
pyzmq                              20.0.0
QDarkStyle                         2.8.1
QtAwesome                          1.0.2
qtconsole                          5.0.3
QtPy                               1.9.0
regex                              2021.4.4
requests                           2.25.1
requests-oauthlib                  1.3.0
rope                               0.18.0
rsa                                4.7.2
Rtree                              0.9.7
ruamel-yaml-conda                  0.15.100
scikit-image                       0.18.1
scikit-learn                       0.24.1
scipy                              1.6.2
seaborn                            0.11.1
Send2Trash                         1.5.0
setuptools                         52.0.0.post20210125
simplegeneric                      0.8.1
singledispatch                     0.0.0
sip                                4.19.13
six                                1.15.0
sniffio                            1.2.0
snowballstemmer                    2.1.0
sortedcollections                  2.1.0
sortedcontainers                   2.3.0
soupsieve                          2.2.1
Sphinx                             4.0.1
sphinxcontrib-applehelp            1.0.2
sphinxcontrib-devhelp              1.0.2
sphinxcontrib-htmlhelp             1.0.3
sphinxcontrib-jsmath               1.0.1
sphinxcontrib-qthelp               1.0.3
sphinxcontrib-serializinghtml      1.1.4
sphinxcontrib-websupport           1.2.4
spyder                             4.2.5
spyder-kernels                     1.10.2
SQLAlchemy                         1.4.7
statsmodels                        0.12.2
sympy                              1.8
tables                             3.6.1
tblib                              1.7.0
tensorboard                        2.6.0
tensorboard-data-server            0.6.1
tensorboard-plugin-wit             1.8.0
tensorflow                         2.6.0
tensorflow-estimator               2.6.0
termcolor                          1.1.0
terminado                          0.9.4
testpath                           0.4.4
textdistance                       4.2.1
threadpoolctl                      2.1.0
three-merge                        0.1.1
tifffile                           2021.4.8
toml                               0.10.2
toolz                              0.11.1
tornado                            6.1
tqdm                               4.59.0
traitlets                          5.0.5
typed-ast                          1.4.2
typing-extensions                  3.7.4.3
ujson                              4.0.2
unicodecsv                         0.14.1
urllib3                            1.26.4
watchdog                           1.0.2
wcwidth                            0.2.5
webencodings                       0.5.1
Werkzeug                           1.0.1
wheel                              0.36.2
widgetsnbextension                 3.5.1
win-inet-pton                      1.1.0
win-unicode-console                0.5
wincertstore                       0.2
wrapt                              1.12.1
xlrd                               2.0.1
XlsxWriter                         1.3.8
xlwings                            0.23.0
xlwt                               1.3.0
xmltodict                          0.12.0
yapf                               0.31.0
yfinance                           0.1.63
zict                               2.0.0
zipp                               3.4.1
zope.event                         4.5.0
zope.interface                     5.3.0

```
Then edit the file "stock_prediction_deep_learning.py" to include the Stock you want to use and the relevant dates and execute:

```cmd
python stock_prediction_deep_learning.py
```

# 5) CUDA installation

![image](https://user-images.githubusercontent.com/7347994/136704184-da189b61-a3b0-4ac8-bd96-3cf669e25823.png)

As I mentioned before, I'm using a GPU to ramp up my testing. As my laptop has a nvidia geforce card, I installed CUDA to make use of its GPU capabilities. Depending on your tensorflow version you'll need a version or another.
Here is the link: https://developer.nvidia.com/cuda-11.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal

You can do from your conda prompt:

```bash
(base) >conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
Collecting package metadata (current_repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.
Collecting package metadata (repodata.json): done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 4.10.3
  latest version: 4.14.0

Please update conda by running

    $ conda update -n base -c defaults conda



## Package Plan ##

  environment location: C:\Users\jordi\anaconda3

  added / updated specs:
    - cudatoolkit=11.2
    - cudnn=8.1.0


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    conda-4.12.0               |   py39hcbf5309_0         1.0 MB  conda-forge
    cudatoolkit-11.2.2         |      h933977f_10       879.9 MB  conda-forge
    cudnn-8.1.0.77             |       h3e0f4f4_0       610.8 MB  conda-forge
    python_abi-3.9             |           2_cp39           4 KB  conda-forge
    ------------------------------------------------------------
                                           Total:        1.46 GB

The following NEW packages will be INSTALLED:

  cudatoolkit        conda-forge/win-64::cudatoolkit-11.2.2-h933977f_10
  cudnn              conda-forge/win-64::cudnn-8.1.0.77-h3e0f4f4_0
  python_abi         conda-forge/win-64::python_abi-3.9-2_cp39

The following packages will be UPDATED:

  conda              pkgs/main::conda-4.10.3-py39haa95532_0 --> conda-forge::conda-4.12.0-py39hcbf5309_0


Proceed ([y]/n)? y


Downloading and Extracting Packages
conda-4.12.0         | 1.0 MB    | ############################################################################ | 100%
python_abi-3.9       | 4 KB      | ############################################################################ | 100%
cudnn-8.1.0.77       | 610.8 MB  | ############################################################################ | 100%
cudatoolkit-11.2.2   | 879.9 MB  | ############################################################################ | 100%
Preparing transaction: done
Verifying transaction: done
Executing transaction: / "By downloading and using the CUDA Toolkit conda packages, you accept the terms and conditions of the CUDA End User License Agreement (EULA): https://docs.nvidia.com/cuda/eula/index.html"

| "By downloading and using the cuDNN conda packages, you accept the terms and conditions of the NVIDIA cuDNN EULA - https://docs.nvidia.com/deeplearning/cudnn/sla/index.html"

done
```

If you run the project after this, the GPU should be correctly picked up:

![image](https://user-images.githubusercontent.com/7347994/190925144-a1b5d934-683f-4d43-a083-fd7c927ed6c5.png)

# 6) Graphviz installation

if you see this message, then you need to install GraphViz library:

```bash
You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) for plot_model/model_to_dot to work.
```

- https://graphviz.gitlab.io/download/

## Sponsors
No sponsors yet! Will you be the first?

[![PayPayl donate button](https://img.shields.io/badge/paypal-donate-yellow.svg)](https://www.paypal.com/paypalme/jordicorbilla)
