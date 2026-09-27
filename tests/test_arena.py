import pandas as pd
import pytest

from quant_forecast_lab.arena import arena_frame, render_html_report


def test_arena_groups_assets_and_ranks_forecasts(tmp_path):
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 5 + ["BBB"] * 5,
            "date": [5, 1, 4, 2, 3, 5, 1, 4, 2, 3],
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
        time_col="date",
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


def test_multi_horizon_requires_origin_close():
    frame = pd.DataFrame({
        "Horizon": [3, 3, 3],
        "Actual": [110.0, 111.0, 112.0],
        "Forecast": [109.0, 110.0, 113.0],
        "OriginClose": [100.0, 101.0, 102.0],
    })
    with pytest.raises(ValueError, match="forecast origin"):
        arena_frame(frame, actual_col="Actual", prediction_cols=["Forecast"], group_cols=["Horizon"])
    result = arena_frame(
        frame, actual_col="Actual", prediction_cols=["Forecast"],
        group_cols=["Horizon"], previous_actual_col="OriginClose",
    )
    assert result.loc[0, "observations"] == 3
