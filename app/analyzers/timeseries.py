import logging
from typing import Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

MIN_ROWS_FOR_TIMESERIES = 20


def detect_time_columns(df: pl.DataFrame, column_types: dict) -> list[str]:
    """Find columns that look like timestamps or dates."""
    time_cols = []
    for col, info in column_types.items():
        if info.get("detected_type") == "datetime":
            time_cols.append(col)
    return time_cols


def try_parse_dates(series: pl.Series) -> Optional[pl.Series]:
    """Attempt to parse a string column as dates."""
    if series.dtype in (pl.Date, pl.Datetime):
        return series

    if series.dtype != pl.Utf8:
        return None

    formats = ["%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S",
               "%m/%d/%Y %H:%M:%S", "%Y/%m/%d"]

    for fmt in formats:
        try:
            parsed = series.str.strptime(pl.Date, fmt, strict=False)
            null_pct = parsed.null_count() / max(len(parsed), 1)
            if null_pct < 0.3:
                return parsed
        except Exception:
            continue

    return None


def check_seasonality(values: np.ndarray, max_lag: int = 50) -> dict:
    """Simple autocorrelation-based seasonality check."""
    n = len(values)
    if n < max_lag * 2:
        max_lag = n // 4

    if max_lag < 2:
        return {"detected": False, "period": None}

    mean = np.nanmean(values)
    var = np.nanvar(values)
    if var == 0:
        return {"detected": False, "period": None}

    autocorrs = []
    for lag in range(1, max_lag + 1):
        c = np.nanmean((values[:-lag] - mean) * (values[lag:] - mean)) / var
        autocorrs.append(float(c))

    # find peaks in autocorrelation
    threshold = 0.3
    peaks = []
    for i in range(1, len(autocorrs) - 1):
        if (autocorrs[i] > autocorrs[i - 1] and
                autocorrs[i] > autocorrs[i + 1] and
                autocorrs[i] > threshold):
            peaks.append({"lag": i + 1, "autocorr": round(autocorrs[i], 4)})

    if peaks:
        best = max(peaks, key=lambda p: p["autocorr"])
        return {"detected": True, "period": best["lag"], "strength": best["autocorr"]}

    return {"detected": False, "period": None}


def detect_trend(values: np.ndarray) -> dict:
    """Fit linear regression to detect trend."""
    n = len(values)
    if n < 5:
        return {"direction": "none", "slope": 0}

    x = np.arange(n, dtype=np.float64)
    valid = ~np.isnan(values)
    if valid.sum() < 5:
        return {"direction": "none", "slope": 0}

    x_clean = x[valid]
    y_clean = values[valid].astype(np.float64)

    # simple OLS
    x_mean = x_clean.mean()
    y_mean = y_clean.mean()
    ss_xy = ((x_clean - x_mean) * (y_clean - y_mean)).sum()
    ss_xx = ((x_clean - x_mean) ** 2).sum()

    if ss_xx == 0:
        return {"direction": "none", "slope": 0}

    slope = ss_xy / ss_xx
    # normalize slope relative to mean
    normalized = slope / abs(y_mean) if y_mean != 0 else slope

    if normalized > 0.001:
        direction = "increasing"
    elif normalized < -0.001:
        direction = "decreasing"
    else:
        direction = "stable"

    return {
        "direction": direction,
        "slope": round(float(slope), 6),
        "normalized_slope": round(float(normalized), 6),
    }


def analyze_timeseries(df: pl.DataFrame, column_types: dict) -> dict:
    """Run time series analysis if applicable."""
    logger.info("Checking for time series patterns")

    time_cols = detect_time_columns(df, column_types)
    numeric_cols = [col for col, info in column_types.items()
                    if info.get("detected_type") == "numeric"]

    if not time_cols:
        logger.info("No time columns detected, skipping time series analysis")
        return {"has_time_component": False}

    if df.shape[0] < MIN_ROWS_FOR_TIMESERIES:
        return {"has_time_component": True, "note": "too few rows for meaningful analysis"}

    results = {
        "has_time_component": True,
        "time_columns": time_cols,
        "analyses": {},
    }

    # pick the first time column as primary
    primary_time = time_cols[0]

    for num_col in numeric_cols[:10]:  # limit columns analyzed
        values = df[num_col].drop_nulls().to_numpy().astype(np.float64)
        if len(values) < MIN_ROWS_FOR_TIMESERIES:
            continue

        trend = detect_trend(values)
        seasonality = check_seasonality(values)

        results["analyses"][num_col] = {
            "time_column": primary_time,
            "trend": trend,
            "seasonality": seasonality,
        }

    n_analyzed = len(results["analyses"])
    logger.info(f"Time series analysis complete for {n_analyzed} columns")
    return results
