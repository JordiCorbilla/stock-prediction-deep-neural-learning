# Copyright 2020-2026 Jordi Corbilla. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
"""Small yfinance market-data inspection utility."""

import argparse
import datetime

import pandas as pd
import yfinance as yf


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download market data for one or more tickers.")
    parser.add_argument("tickers", nargs="*", default=["ETH-USD", "GOOG", "META", "TSLA"])
    parser.add_argument("--start-date", default="2004-08-01")
    args = parser.parse_args(argv)

    start = pd.to_datetime(args.start_date)
    for ticker in args.tickers:
        print(f"\n=== {ticker} ===")
        data = yf.download(
            ticker,
            start=start,
            end=datetime.date.today(),
            auto_adjust=False,
            progress=False,
        )
        print(data)


if __name__ == "__main__":
    main()
