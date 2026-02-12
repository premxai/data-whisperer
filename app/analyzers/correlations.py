import logging
from itertools import combinations

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

HIGH_CORR_THRESHOLD = 0.7


def pearson_correlation_matrix(df: pl.DataFrame, numeric_cols: list[str]) -> dict:
    """Compute Pearson correlation matrix for numeric columns."""
    if len(numeric_cols) < 2:
        return {"matrix": {}, "high_correlations": []}

    subset = df.select(numeric_cols).fill_null(strategy="mean")

    try:
        matrix = subset.to_numpy().astype(np.float64)
    except Exception as e:
        logger.warning(f"Correlation matrix failed: {e}")
        return {"matrix": {}, "high_correlations": []}

    # handle columns with zero variance
    stds = np.nanstd(matrix, axis=0)
    valid_mask = stds > 0
    valid_cols = [c for c, v in zip(numeric_cols, valid_mask) if v]
    matrix = matrix[:, valid_mask]

    if matrix.shape[1] < 2:
        return {"matrix": {}, "high_correlations": []}

    corr = np.corrcoef(matrix, rowvar=False)

    # build dict representation
    corr_dict = {}
    for i, col_i in enumerate(valid_cols):
        corr_dict[col_i] = {}
        for j, col_j in enumerate(valid_cols):
            val = float(corr[i, j])
            if np.isnan(val):
                val = 0.0
            corr_dict[col_i][col_j] = round(val, 4)

    # find high correlations
    high = []
    for (i, col_i), (j, col_j) in combinations(enumerate(valid_cols), 2):
        val = corr[i, j]
        if not np.isnan(val) and abs(val) >= HIGH_CORR_THRESHOLD:
            high.append({
                "col1": col_i,
                "col2": col_j,
                "correlation": round(float(val), 4),
            })

    high.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    return {
        "matrix": corr_dict,
        "high_correlations": high,
    }


def cramers_v(series_a: pl.Series, series_b: pl.Series) -> float:
    """Compute Cramer's V statistic for two categorical columns."""
    a = series_a.drop_nulls().to_list()
    b = series_b.drop_nulls().to_list()

    # align lengths (use only rows where both are non-null)
    mask_a = series_a.is_not_null()
    mask_b = series_b.is_not_null()
    combined_mask = mask_a & mask_b

    a = series_a.filter(combined_mask).to_list()
    b = series_b.filter(combined_mask).to_list()

    if len(a) < 2:
        return 0.0

    # build contingency table
    pairs = list(zip(a, b))
    a_vals = sorted(set(a))
    b_vals = sorted(set(b))

    if len(a_vals) < 2 or len(b_vals) < 2:
        return 0.0

    a_map = {v: i for i, v in enumerate(a_vals)}
    b_map = {v: i for i, v in enumerate(b_vals)}

    table = np.zeros((len(a_vals), len(b_vals)))
    for av, bv in pairs:
        table[a_map[av], b_map[bv]] += 1

    n = table.sum()
    if n == 0:
        return 0.0

    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)
    expected = np.outer(row_sums, col_sums) / n

    # avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.where(expected > 0, (table - expected) ** 2 / expected, 0).sum()

    k = min(len(a_vals), len(b_vals))
    if k <= 1 or n <= 1:
        return 0.0

    v = np.sqrt(chi2 / (n * (k - 1)))
    return round(float(min(v, 1.0)), 4)


def categorical_correlations(df: pl.DataFrame, categorical_cols: list[str]) -> dict:
    """Compute Cramer's V for all pairs of categorical columns."""
    if len(categorical_cols) < 2:
        return {"pairs": []}

    # limit to avoid combinatorial explosion
    cols = categorical_cols[:15]

    pairs = []
    for col_a, col_b in combinations(cols, 2):
        v = cramers_v(df[col_a], df[col_b])
        if v > 0.1:  # only report meaningful associations
            pairs.append({
                "col1": col_a,
                "col2": col_b,
                "cramers_v": v,
            })

    pairs.sort(key=lambda x: x["cramers_v"], reverse=True)
    return {"pairs": pairs}


def analyze_correlations(df: pl.DataFrame, column_types: dict) -> dict:
    """Run all correlation analyses."""
    logger.info("Running correlation analysis")

    numeric_cols = [col for col, info in column_types.items()
                    if info.get("detected_type") == "numeric"]
    categorical_cols = [col for col, info in column_types.items()
                        if info.get("detected_type") == "categorical"]

    pearson = pearson_correlation_matrix(df, numeric_cols)
    categorical = categorical_correlations(df, categorical_cols)

    n_high = len(pearson["high_correlations"])
    n_cat_pairs = len(categorical["pairs"])
    logger.info(f"Correlation analysis done. {n_high} high numeric, {n_cat_pairs} categorical associations")

    return {
        "numeric": pearson,
        "categorical": categorical,
    }
