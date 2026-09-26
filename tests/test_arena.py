import pandas as pd

from quant_forecast_lab.arena import arena_frame, render_html_report


def test_arena_groups_assets_and_ranks_forecasts(tmp_path):
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 5 + ["BBB"] * 5,
            "actual": [100, 101, 99, 103, 102, 50, 51, 50, 52, 53],
            "perfect": [100, 101, 99, 103, 102, 50, 51, 50, 52, 53],
            "lagged": [100, 100, 101, 99, 103, 50, 50, 51, 50, 52],
        }
    )

    result = arena_frame(
        frame,
        actual_col="actual",
        prediction_cols=["perfect", "lagged"],
        group_cols=["ticker"],
        transaction_cost_bps=5,
    )

    assert set(result["ticker"]) == {"AAA", "BBB"}
    for ticker in ("AAA", "BBB"):
        group = result[result["ticker"] == ticker]
        assert group.iloc[0]["model"] == "perfect"
        assert group.iloc[0]["rmse_skill_vs_naive"] == 1.0

    report = render_html_report(result, tmp_path / "arena.html")
    assert report.exists()
    assert "Financial Forecasting Arena" in report.read_text(encoding="utf-8")
