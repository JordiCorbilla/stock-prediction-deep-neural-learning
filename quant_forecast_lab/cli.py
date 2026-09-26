"""Command-line research utilities that preserve legacy training entry points."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from .benchmark import benchmark_frame


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant-forecast")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Compare aligned price predictions with a last-price naive baseline.",
    )
    benchmark.add_argument("--csv", required=True)
    benchmark.add_argument("--actual-col", required=True)
    benchmark.add_argument(
        "--prediction-col",
        action="append",
        dest="prediction_cols",
        required=True,
    )
    benchmark.add_argument("--previous-actual-col")
    benchmark.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "benchmark":
        frame = pd.read_csv(args.csv)
        result = benchmark_frame(
            frame,
            actual_col=args.actual_col,
            prediction_cols=args.prediction_cols,
            previous_actual_col=args.previous_actual_col,
        )
        if args.as_json:
            print(json.dumps(result.reset_index().to_dict(orient="records"), indent=2))
        else:
            print(result.to_string(float_format=lambda value: f"{value:.6f}"))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
