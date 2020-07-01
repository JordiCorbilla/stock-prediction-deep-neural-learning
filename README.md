# Stock prediction using deep neural learning

Predicting stock prices using a TensorFlow LSTM (long short-term memory) neural network for times series forecasting

## Introduction

Predicting stock prices is a cumbersome task as it does not follow any specific pattern. Changes in the stock prices are purely based on buy/sell actions during a period of time. In order to learn the specific characteristics of a stock price, we can use deep learning to identify these patterns through learning. One of the most well-known networks for series forecasting is LSTM (long short-term memory) which is a recurrent neural network that is able to remember information over a long period of time, thus making them extremely useful for predicting stock prices.

## Stock Market Data

The initial data we will use for this model is taken directly from the [Yahoo Finance](https://finance.yahoo.com/quote/GOOG?p=GOOG) page which contains the latest market data on a specific stock price. To perform this operation easily using Python, we will use the [yFinance](https://aroussi.com/post/python-yahoo-finance) library which has been built specifically for this and that it will allow us to download all the information we need on a given [ticker symbol](https://www.investopedia.com/terms/t/tickersymbol.asp).

