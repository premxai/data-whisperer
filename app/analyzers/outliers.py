import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

IQR_MULTIPLIER = 1.5


def iqr_outliers(series: pl.Series) -> dict:
    """Detect outliers using the IQR method for a single numeric column."""
    clean = series.drop_nulls().cast(pl.Float64)
    if len(clean) < 4:
        return {"count": 0, "indices": [], "bounds": None}

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1

    lower = q1 - IQR_MULTIPLIER * iqr
    upper = q3 + IQR_MULTIPLIER * iqr

    mask = (clean < lower) | (clean > upper)
    outlier_indices = [i for i, v in enumerate(mask.to_list()) if v]

    return {
        "count": len(outlier_indices),
        "pct": round(len(outlier_indices) / len(clean) * 100, 2),
        "indices": outlier_indices[:100],  # cap to avoid huge lists
        "bounds": {"lower": round(lower, 4), "upper": round(upper, 4)},
    }


def isolation_forest_outliers(df: pl.DataFrame, numeric_cols: list[str],
                               contamination: float = 0.05) -> dict:
    """Multivariate outlier detection using Isolation Forest."""
    from sklearn.ensemble import IsolationForest

    if len(numeric_cols) < 2:
        return {"count": 0, "indices": [], "note": "need at least 2 numeric columns"}

    # build matrix from numeric columns, fill nulls with median
    subset = df.select(numeric_cols)
    filled = subset.fill_null(strategy="mean")

    try:
        matrix = filled.to_numpy().astype(np.float64)
    except Exception as e:
        logger.warning(f"Could not convert to numpy: {e}")
        return {"count": 0, "indices": [], "note": str(e)}

    # drop rows that are still nan after fill
    valid_mask = ~np.isnan(matrix).any(axis=1)
    if valid_mask.sum() < 10:
        return {"count": 0, "indices": [], "note": "not enough valid rows"}

    clean_matrix = matrix[valid_mask]

    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    preds = iso.fit_predict(clean_matrix)

    # map back to original indices
    original_indices = np.where(valid_mask)[0]
    outlier_indices = original_indices[preds == -1].tolist()

    return {
        "count": len(outlier_indices),
        "pct": round(len(outlier_indices) / len(df) * 100, 2),
        "indices": outlier_indices[:200],
    }


def detect_outliers(df: pl.DataFrame, column_types: dict) -> dict:
    """Run outlier detection on all numeric columns."""
    logger.info("Running outlier detection")

    numeric_cols = [col for col, info in column_types.items()
                    if info.get("detected_type") == "numeric"]

    if not numeric_cols:
        logger.info("No numeric columns found, skipping outlier detection")
        return {"univariate": {}, "multivariate": {}}

    # per-column IQR
    univariate = {}
    for col in numeric_cols:
        result = iqr_outliers(df[col])
        if result["count"] > 0:
            univariate[col] = result

    # multivariate
    multivariate = isolation_forest_outliers(df, numeric_cols)

    total_flagged = sum(r["count"] for r in univariate.values())
    logger.info(f"Outlier detection complete. {total_flagged} univariate, "
                f"{multivariate['count']} multivariate outliers found")

    return {
        "univariate": univariate,
        "multivariate": multivariate,
    }
