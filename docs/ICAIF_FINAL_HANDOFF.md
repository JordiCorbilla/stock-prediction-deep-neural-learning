# ICAIF v9 handoff — November 1, 2026

The maintained default branch is `master`. The ICAIF adapter/trainer and the
completed FOMC research branch have been merged here. The architecture remains
Jordi Corbilla's `stock_prediction_lstm.py`, v9 shared 128/64 LSTM encoder with
the expected-return head. Its SHA-256 is
`2579877f487ed6e7b311804c56c47a66adf61955c6223e055d43bbada50aabd7`.

## Code and evidence map

- `stock_prediction_lstm.py`: original architecture/factory.
- `quant_forecast_lab/icaif.py`: canonical causal features and portfolio targets;
  the frozen arithmetic order is the default.
- `quant_forecast_lab/icaif_training.py`: three-seed training through that factory;
  run from a source checkout, using the documented purged samples.
- `quant_forecast_lab/icaif_fomc.py`: separately documented research extension,
  not part of the submitted model. Its later-year promotion gate failed.
- [Adapter methodology and historical experiments](ICAIF_COMPETITION_ADAPTER.md).
- [FOMC experiment and rejected promotion](ICAIF_FOMC_RESEARCH.md).
- [Fresh offline reproduction result](evidence/icaif-final-replay-2026-11-01.json).

The competition API client, portfolio accounting, sealed decisions and private
credentials belong to the sibling
`JordiCorbilla/acm-icaif-2026-trading-agent-competition` repository. They are not
required to replay the public demonstration. Models and datasets are supplied
in the frozen review package, rather than committed as new Git artifacts here.

## Frozen reproducibility package

Final submission **2131** was uploaded once on October 31. Its owned receipt
was `VALID`, `PENDING_REVIEW` on November 1; final ranking/review is not confirmed.
Official trading decision 1545 is the earlier decision reproduced by this package.

[Shared materials folder](https://drive.google.com/drive/folders/117InGe2pyjgp1tcHjHapc-ehwGXsMCP5?usp=sharing)
contains `source.zip`, the video, README, disclosures, dependencies and checksums.
The canonical `source.zip` SHA-256 is
`aeca6ce28c755486e53c14b2713ac09c0b33bbe7a75093dc583a45e3db5183d6`.
Do not replace it with a new archive of today's default branch.

Download `source.zip`, check that hash, and extract it to a dedicated folder.
From the extracted root, follow its pinned Python 3.11/Windows setup:

```powershell
py -3.11 -m venv ..\icaif-review-env
& ..\icaif-review-env\Scripts\python.exe -m pip install -r requirements.txt
& ..\icaif-review-env\Scripts\python.exe -B reproduce.py --verify-only
& ..\icaif-review-env\Scripts\python.exe -B reproduce.py --output ..\icaif-review-output
```

The output directory must be new and outside the extracted package. Replay
needs no account credentials, data downloads or trading submissions. Package
installation needs package-index access. Frozen checkpoints are canonical;
new training is unnecessary and is not claimed to yield identical weights.

## November 1 checks

The full unit/runtime suite passed in the existing LSTM environment, together
with lint, compilation, dependency consistency and training CLI help checks.
The competition suite passed all 138 tests. Lineage checks verified 34 archived
models and the frozen live checkpoint hashes. Current canonical features match
the stored 2023 train/validation/test tensors and the Official 30×20×6 input
bit for bit. An independent replay using the pinned clean environment verified
188 packaged files and reproduced the three forecasts, ensemble and all weights.
Legacy v7 compatibility checks emit existing scaler-version/deprecation warnings;
those checks do not establish v7 compatibility across arbitrary runtimes.

An upstream running-sum optimization changed feature rounding by up to roughly
1e-6 on the checked historical tensors. It remains available explicitly as
`enhanced_features_fast` for research; the default `enhanced_features` retains
the exact frozen calculation. A small historical tensor fixture guards this.
Neither frozen models nor cloud review materials were modified.

All remaining ICAIF automations were deleted at the user's request. No scheduled
monitoring or upload is needed. Keep shared materials accessible and unchanged
through organizer review; respond manually if organizers request clarification.
