import logging
import re
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")

# thresholds for type detection
CATEGORICAL_THRESHOLD = 0.05  # if unique/total < this, likely categorical
MAX_CATEGORICAL_UNIQUE = 50


def detect_column_type(series: pl.Series) -> str:
    """Classify a column as numeric, categorical, datetime, or text."""
    dtype = series.dtype

    if dtype in (pl.Date, pl.Datetime, pl.Time):
        return "datetime"

    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                 pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                 pl.Float32, pl.Float64):
        n_unique = series.n_unique()
        if n_unique <= MAX_CATEGORICAL_UNIQUE and n_unique / max(len(series), 1) < CATEGORICAL_THRESHOLD:
            return "categorical"
        return "numeric"

    if dtype == pl.Utf8:
        # try parsing as dates
        non_null = series.drop_nulls()
        if len(non_null) > 0:
            sample = non_null.head(20).to_list()
            if _looks_like_dates(sample):
                return "datetime"

        n_unique = series.n_unique()
        if n_unique <= MAX_CATEGORICAL_UNIQUE:
            return "categorical"
        return "text"

    if dtype == pl.Boolean:
        return "categorical"

    return "text"


def _looks_like_dates(values: list) -> bool:
    date_patterns = [
        re.compile(r"\d{4}-\d{2}-\d{2}"),
        re.compile(r"\d{2}/\d{2}/\d{4}"),
        re.compile(r"\d{2}-\d{2}-\d{4}"),
    ]
    matches = 0
    for v in values:
        s = str(v).strip()
        if any(p.match(s) for p in date_patterns):
            matches += 1
    return matches / max(len(values), 1) > 0.8


def profile_numeric(series: pl.Series) -> dict:
    clean = series.drop_nulls()
    if len(clean) == 0:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}

    return {
        "mean": round(float(clean.mean()), 4),
        "median": round(float(clean.median()), 4),
        "std": round(float(clean.std()), 4),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "q25": round(float(clean.quantile(0.25)), 4),
        "q75": round(float(clean.quantile(0.75)), 4),
    }


def profile_categorical(series: pl.Series) -> dict:
    counts = series.value_counts().sort("count", descending=True)
    top_values = []
    for row in counts.head(10).iter_rows():
        top_values.append({"value": str(row[0]), "count": int(row[1])})

    return {
        "n_unique": series.n_unique(),
        "top_values": top_values,
    }


def detect_pii(series: pl.Series) -> list[str]:
    """Check string columns for potential PII patterns."""
    flags = []
    sample = series.drop_nulls().head(100).to_list()

    email_hits = sum(1 for v in sample if EMAIL_PATTERN.search(str(v)))
    if email_hits > len(sample) * 0.3:
        flags.append("potential_email")

    phone_hits = sum(1 for v in sample if PHONE_PATTERN.search(str(v)))
    if phone_hits > len(sample) * 0.3:
        flags.append("potential_phone")

    return flags


def profile_column(series: pl.Series) -> dict[str, Any]:
    col_type = detect_column_type(series)
    total = len(series)
    null_count = series.null_count()

    result = {
        "name": series.name,
        "detected_type": col_type,
        "dtype": str(series.dtype),
        "total_count": total,
        "null_count": null_count,
        "null_pct": round(null_count / max(total, 1) * 100, 2),
        "n_unique": series.n_unique(),
    }

    if col_type == "numeric":
        result["stats"] = profile_numeric(series)
    elif col_type == "categorical":
        result["stats"] = profile_categorical(series)
    elif col_type == "text":
        result["stats"] = profile_categorical(series)
        result["pii_flags"] = detect_pii(series)
    elif col_type == "datetime":
        clean = series.drop_nulls()
        if len(clean) > 0:
            result["stats"] = {"min": str(clean.min()), "max": str(clean.max())}

    return result


def profile_dataset(df: pl.DataFrame) -> dict:
    """Generate a full profile of the dataset."""
    logger.info(f"Profiling dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    columns = {}
    type_counts = {"numeric": 0, "categorical": 0, "datetime": 0, "text": 0}
    total_nulls = 0

    for col in df.columns:
        col_profile = profile_column(df[col])
        columns[col] = col_profile
        type_counts[col_profile["detected_type"]] += 1
        total_nulls += col_profile["null_count"]

    total_cells = df.shape[0] * df.shape[1]
    completeness = round((1 - total_nulls / max(total_cells, 1)) * 100, 2)

    # check for duplicate rows
    n_duplicates = df.shape[0] - df.unique().shape[0]

    profile = {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "type_counts": type_counts,
        "completeness_pct": completeness,
        "total_nulls": total_nulls,
        "duplicate_rows": n_duplicates,
        "columns": columns,
    }

    logger.info(f"Profile complete. Completeness: {completeness}%, Duplicates: {n_duplicates}")
    return profile
