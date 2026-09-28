# Copyright 2020-2026 Jordi Corbilla. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
"""Inspect the metadata and fundamental datasets exposed by yfinance."""

import argparse
import json

import yfinance as yf


def _print_value(label, getter):
    print(label)
    try:
        value = getter()
        if isinstance(value, dict):
            print(json.dumps(value, indent=2, sort_keys=True, default=str))
        else:
            print(value)
    except Exception as exc:
        print(f"Unavailable: {exc}")
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect yfinance information for a ticker.")
    parser.add_argument("ticker", nargs="?", default="^FTSE")
    args = parser.parse_args(argv)

    sec = yf.Ticker(args.ticker)

    _print_value("Info", lambda: sec.info)
    _print_value("Fast Info", lambda: dict(sec.fast_info))
    _print_value("History", lambda: sec.history(period="1mo"))
    _print_value("Major Holders", lambda: sec.major_holders)
    _print_value("Institutional Holders", lambda: sec.institutional_holders)
    _print_value("Dividends", lambda: sec.dividends)
    _print_value("Splits", lambda: sec.splits)
    _print_value("Actions", lambda: sec.actions)
    _print_value("Calendar", lambda: sec.calendar)
    _print_value("Recommendations", lambda: sec.recommendations)
    _print_value("Income Statement", lambda: sec.income_stmt)
    _print_value("Quarterly Income Statement", lambda: sec.quarterly_income_stmt)
    _print_value("Balance Sheet", lambda: sec.balance_sheet)
    _print_value("Quarterly Balance Sheet", lambda: sec.quarterly_balance_sheet)
    _print_value("Cash Flow", lambda: sec.cashflow)
    _print_value("Quarterly Cash Flow", lambda: sec.quarterly_cashflow)
    _print_value("Sustainability", lambda: sec.sustainability)
    _print_value("Options", lambda: sec.options)


if __name__ == "__main__":
    main()
