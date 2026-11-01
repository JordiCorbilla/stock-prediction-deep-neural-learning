# Changelog

## 2.1.0 — 2026-11-01

### Added
- Merged ICAIF v9 causal features, three-seed training entrypoint, portfolio
  adapter and documented experiment provenance into the default branch.
- Included completed, unpromoted FOMC calendar research and its tests.
- Added a final handoff guide, offline replay evidence and a frozen tensor
  regression fixture.

### Fixed
- Restored the frozen feature arithmetic as the default to preserve bitwise
  compatibility with trained ICAIF inputs. The running-sum implementation is
  retained explicitly as `enhanced_features_fast` for research.
- Aligned package, module and citation version metadata with the repository's
  v2 release series.

### Validation and scope
- Unit/runtime tests, lint, compilation and installed dependency checks pass.
- All 138 sibling competition tests pass; frozen model lineage and offline
  forecasts, ensemble and execution weights are verified.
- This is a source release checkpoint, not a claim of guaranteed returns,
  completed organizer review or universal cross-platform reproducibility.
- Frozen submitted source/model materials remain unchanged.

## Earlier architecture work — included in this release

### Added
- Maintained quant_forecast_lab research utility package.
- Naive-relative forecast evaluation.
- Expanding-window validation primitives.
- Conformal interval utilities.
- Cost-aware directional backtest diagnostics.
- CSV benchmark CLI.
- Unit tests and GitHub Actions CI.
- Contributor and issue templates.

### Changed
- Repository layout cleaned so generated experiments no longer dominate the project root.
- Current source distribution uses Apache-2.0. Earlier CC0 distributions remain governed by their original terms.
- Documentation reorganised around reproducibility and out-of-sample evaluation.

### Compatibility
- Historical root training and inference scripts are retained.
- Existing committed experiment artifacts remain available in Git history.
