"""Causal ICAIF features and execution gates for Jordi Corbilla's v9 LSTM.

These adapters change inputs and portfolio decisions, not the LSTM architecture.
"""

import numpy as np

FEATURE_VERSION = "icaif-causal-v1"


def enhanced_features(base):
    """Canonical prefix features with the frozen ICAIF arithmetic order.

    Keep reduction order unchanged: the submitted checkpoints were fitted on
    these exact float32 tensors. Use enhanced_features_fast explicitly for
    research that tolerates different rounding.
    """
    x = np.asarray(base)
    if x.ndim != 3 or x.shape[2] != 3 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite stock/time/three-feature input")
    returns = x[:, :, 0].astype(float) * .02
    if (returns <= -1).any():
        raise ValueError("Invalid stock return")
    logs = np.cumsum(np.log1p(returns), axis=1)
    bands = np.zeros_like(returns)
    acceleration = np.zeros_like(returns)
    for t in range(2, returns.shape[1]):
        time = np.arange(t + 1)
        centered = time - time.mean()
        prefix = logs[:, :t + 1]
        slope = prefix @ centered / (centered @ centered)
        fitted = prefix.mean(axis=1) + slope * (t - time.mean())
        volatility = np.maximum(returns[:, :t + 1].std(axis=1, ddof=1), 1e-4)
        bands[:, t] = np.clip((logs[:, t] - fitted) / volatility, -5, 5) / 5
        if t >= 9:
            acceleration[:, t] = returns[:, t - 4:t + 1].sum(axis=1) - returns[:, t - 9:t - 4].sum(axis=1)
    return np.concatenate([x, logs[:, :, None] / .02, bands[:, :, None],
                           acceleration[:, :, None] / .02], axis=2).astype("float32")


def enhanced_features_fast(base):
    """Vectorized research alternative; not bitwise compatible with frozen inputs."""
    x = np.asarray(base)
    if x.ndim != 3 or x.shape[2] != 3 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite stock/time/three-feature input")
    returns = x[:, :, 0].astype(float) * .02
    if (returns <= -1).any():
        raise ValueError("Invalid stock return")
    logs = np.cumsum(np.log1p(returns), axis=1)
    result = np.zeros(x.shape[:2] + (6,), dtype="float32")
    result[:, :, :3] = x.astype(float, copy=False)
    result[:, :, 3] = logs / .02
    # Prefix regression and ddof=1 volatility for every t >= 2 from running sums.
    time = np.arange(returns.shape[1])
    t = time[2:]
    count = t + 1.
    sum_logs = logs.cumsum(axis=1)[:, 2:]
    sum_time_logs = (logs * time).cumsum(axis=1)[:, 2:]
    slope = (sum_time_logs - t / 2 * sum_logs) / (t * count * (t + 2) / 12)
    fitted = sum_logs / count + slope * (t / 2)
    shifted = returns - returns[:, :1]
    sum_shifted = shifted.cumsum(axis=1)[:, 2:]
    variance = ((shifted ** 2).cumsum(axis=1)[:, 2:] - sum_shifted ** 2 / count) / t
    volatility = np.sqrt(np.maximum(variance, 1e-8))
    result[:, 2:, 4] = np.clip((logs[:, 2:] - fitted) / volatility, -5, 5) / 5
    # Acceleration: second 5-step difference of one zero-padded running sum.
    running = np.cumsum(np.concatenate([np.zeros((len(returns), 1)), returns], axis=1), axis=1)
    result[:, 9:, 5] = (running[:, 10:] - 2 * running[:, 5:-5] + running[:, :-10]) / .02
    return result


def sequence_features(base, lookback=20, vix=None):
    """Recompute causal features on the selected history; append observed VIX.

    VIX contains lookback+1 positive completed closes, aligned to the stock
    history. Fixed scales avoid a scaler fitted on evaluation data.
    """
    if lookback not in (10, 15, 20):
        raise ValueError("Lookback must be 10, 15 or 20")
    x = np.asarray(base)
    if x.ndim != 3 or x.shape[1] < lookback:
        raise ValueError("Insufficient base history")
    result = enhanced_features(x[:, -lookback:, :])
    if vix is not None:
        values = np.asarray(vix, dtype=float)
        if values.shape != (lookback + 1,) or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Expected aligned positive completed VIX closes")
        features = np.column_stack([values[1:] / 40., np.clip(np.diff(np.log(values)) / .1, -5, 5)])
        extra = np.broadcast_to(features, (len(result), lookback, 2))
        result = np.concatenate([result, extra], axis=2).astype("float32")
    return result


def portfolio_target(scores, exposure=.99, tilt=.20):
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (30,) or not np.isfinite(scores).all() or not 0 <= exposure <= .99 or not 0 <= tilt <= .20:
        raise ValueError("Invalid competition target inputs")
    target = np.full(30, exposure * (1 - tilt) / 30)
    winners = np.argsort(-scores, kind="stable")[:10]
    target[winners] += exposure * tilt / 10
    return target


def trade_required(current, target, threshold=.12):
    """Combined buy/sell notional divided by NAV; retain shares when skipped."""
    current = np.asarray(current, dtype=float)
    target = np.asarray(target, dtype=float)
    if current.shape != (30,) or target.shape != (30,) or not np.isfinite(current).all() or not np.isfinite(target).all():
        raise ValueError("Invalid portfolio vectors")
    if (current < 0).any() or current.sum() > 1.000001 or (target < 0).any() or target.sum() > 1.000001 or target.max() > .30:
        raise ValueError("Invalid portfolio constraints")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid no-trade threshold")
    return bool(np.abs(target - current).sum() >= threshold)
