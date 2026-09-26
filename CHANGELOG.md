# Changelog

## Unreleased — architecture-clean-up

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
- Project licensing made consistently Apache-2.0.
- Documentation reorganised around reproducibility and out-of-sample evaluation.

### Compatibility
- Historical root training and inference scripts are retained.
- Existing committed experiment artifacts remain available in Git history.
