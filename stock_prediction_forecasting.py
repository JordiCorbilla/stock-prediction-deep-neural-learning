# Copyright 2020-2026 Jordi Corbilla. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================
"""Compatibility forecasting entry point.

Historically this script attempted to construct StockData without its required
StockPrediction object. It now delegates to the maintained inference runner
while preserving a useful no-argument demo.
"""

import argparse
import os
import secrets
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stock_prediction_deep_learning_inference import InferenceRunner


def build_parser():
    parser = argparse.ArgumentParser(description="Run a saved stock forecast model.")
    parser.add_argument(
        "--run-folder",
        default=os.path.join("examples", "runs", "reference-v7-ftse"),
        help="Folder containing the saved model, scalers and model_config.json.",
    )
    parser.add_argument("--ticker", default="^FTSE")
    parser.add_argument(
        "--output-folder",
        help="Separate writable output directory; defaults to a fresh directory under runs/forecasts/.",
    )
    parser.add_argument("--start-date", default="2017-01-01")
    parser.add_argument("--validation-date", default="2024-03-12")
    parser.add_argument("--forecast-days", type=int, default=30)
    parser.add_argument("--time-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calendar", default="XLON", help="exchange_calendars name, e.g. XLON or XNYS")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    source = Path(args.run_folder).resolve()
    output = Path(args.output_folder).resolve() if args.output_folder else (
        Path('runs') / 'forecasts' / (datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ') + '_' + secrets.token_hex(4))
    ).resolve()
    if output == source or output.is_relative_to(source):
        parser.error('--output-folder must be separate from --run-folder and outside it.')

    runner = InferenceRunner(
        run_folder=args.run_folder,
        ticker=args.ticker,
        start_date=pd.to_datetime(args.start_date),
        validation_date=pd.to_datetime(args.validation_date),
        github_url="",
        epochs=0,
        time_steps=args.time_steps,
        token="forecasting",
        batch_size=1,
        forecast_days=args.forecast_days,
        use_business_days=True,
        plot_history_days=200,
        use_returns=False,
        use_deltas=True,
        clip_negative=True,
        blend_alpha=0.6,
        direction_threshold=0.55,
        mag_clip_pct=90,
        stochastic_paths=50,
        stochastic_seed=args.seed,
        stochastic_sigma_mult=0.6,
        stochastic_lookback=120,
        conformal_coverage=0.90,
        exchange_calendar=args.calendar or None,
        output_folder=str(output),
    )
    return runner.run()


if __name__ == "__main__":
    main()
