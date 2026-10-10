# FOMC context for our ICAIF LSTM

Research extension created October 10, 2026. This keeps Jordi Corbilla's actual
v9 128/64 LSTM encoder and expected-return head. The existing six-feature adapter,
trainer and Official frozen architecture are unchanged.

`quant_forecast_lab/icaif_fomc.py` adds two known-in-advance context channels:
sessions until the next scheduled FOMC announcement, capped at 20 and divided by
20, and a flag for announcement day or the preceding trading session. Context is
for the forecast day and is broadcast across all 20 history steps. This gives
the model the same forecast-day regime at every step; it is not a sequence of
observed announcement outcomes. No policy decision, subsequent headline, future
stock return or future VIX observation enters the feature.

The caller must supply an appropriate ex-ante trading/event calendar. The helper
rejects missing forecast dates and incomplete near-event calendars. It does not
discover a calendar from prices or claim that today's calendar proves historical
publication availability.

The competition repository's `lstm_fomc_research.py` prepares purged chronological
arrays and invokes this repository's existing `quant_forecast_lab.icaif_training`
module. A paired control uses the same eight-channel model with zero-valued
calendar channels. Both use seeds 7, 42, 123, at most 10 epochs and patience 3.
This controls the change in parameter count and random initialization better
than attributing a comparison against the old six-channel model entirely to FOMC.

There is one fixed challenger. It must improve 2023 mean net P&L while meeting
declared downside guards before 2024–2025 confirmation. Both policies use daily
99% equity allocation, an 80% equal-weight anchor and a 20% top-ten tilt, with
0.1% fees on each buy and sell. No leverage or automatic account upload is added.

The target predicts relative stock returns; a common calendar flag needs to
improve cross-sectional selection to affect ranking. Learning a uniform market
return shift alone does not improve this portfolio.

Unit checks cover weekend session counting, the pre-announcement flag, rejection
of missing calendar coverage, and preservation of all original stock features:

```powershell
python -m pytest tests/test_icaif_fomc.py tests/test_icaif.py -q
```

Full protocol, price hashes, model/source hashes, training histories, portfolio
results and promotion decision are maintained in the competition repository's
`docs/LSTM_FOMC_RESEARCH.md` and its dated evidence archive. This research does
not change accepted Official submission 1545.

## Completed outcome

All 18 neural fits completed (validation early stopping after 4–8 epochs).
Mean net P&L per matched 15-session $1m episode:

|Year|Zero-calendar control|FOMC context|Increment|
|---|---:|---:|---:|
|2023|$9,015.96|$9,249.85|+$233.90|
|2024|$12,951.85|$13,106.42|+$154.58|
|2025|$8,370.73|$8,228.27|−$142.46|

The challenger passed the first two annual gates but failed the 2025 profit gate.
It was not promoted. All downside guards passed, but the pooled later-year
increment was only +$6.79 per window, which does not overturn the fixed gate.
These are overlapping historical episodes after 0.1% buy/sell fees, not annual
earnings or a live result. The unchanged Official model remains in use.

See the [full experiment, chart and trade ledger](https://github.com/JordiCorbilla/acm-icaif-2026-trading-agent-competition/blob/main/docs/LSTM_FOMC_RESEARCH.md)
for the protocol, training histories, input/model/source hashes and limitations.
Feature code is published on `codex/fomc-calendar-feature`, preserving concurrent
updates on the existing `architecture-clean-up` branch.
