"""Research-only, known-in-advance FOMC context for the user's ICAIF LSTM."""

from datetime import date

import numpy as np


def calendar_context(forecast_day, sessions, announcements):
    """Bound distance at 20 sessions; same-day and prior-session flag.

    The caller supplies an ex-ante scheduled calendar, never policy outcomes.
    Counts trading sessions rather than calendar days. No prices are inspected.
    """
    day = date.fromisoformat(str(forecast_day)[:10])
    trading = sorted({date.fromisoformat(str(d)[:10]) for d in sessions})
    events = sorted({date.fromisoformat(str(d)[:10]) for d in announcements})
    if day not in trading:
        raise ValueError("Forecast day missing from trading calendar")
    future = [d for d in events if d >= day]
    if not future:
        raise ValueError("Future scheduled FOMC calendar unavailable")
    upcoming = future[0]
    # Far future events need no precise future session calendar beyond the cap.
    within = [d for d in trading if day < d <= upcoming]
    if upcoming not in trading and len(within) < 20:
        raise ValueError("Incomplete trading calendar before next announcement")
    distance = min(len(within), 20)
    return np.array([distance / 20.0, float(distance <= 1)], dtype="float32")


def append_calendar_context(features, context):
    values = np.asarray(features)
    calendar = np.asarray(context, dtype="float32")
    if values.ndim != 3 or values.shape[-1] != 6 or not np.isfinite(values).all():
        raise ValueError("Expected causal six-feature sequences")
    if calendar.shape != (2,) or not np.isfinite(calendar).all() or (calendar < 0).any() or (calendar > 1).any():
        raise ValueError("Invalid calendar context")
    extra = np.broadcast_to(calendar, values.shape[:2] + (2,))
    return np.concatenate([values, extra], axis=-1).astype("float32")
