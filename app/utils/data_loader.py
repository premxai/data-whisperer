import logging
from pathlib import Path
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet"}
MAX_FILE_SIZE_MB = 500


def detect_file_type(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")
    return ext


def load_file(filepath: str, encoding: Optional[str] = None) -> pl.DataFrame:
    """Load a file into a Polars DataFrame. Handles CSV, Excel, JSON, and Parquet."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large: {size_mb:.1f}MB (max {MAX_FILE_SIZE_MB}MB)")

    ext = detect_file_type(filepath)
    logger.info(f"Loading {ext} file: {filepath} ({size_mb:.1f}MB)")

    try:
        if ext == ".csv":
            df = _load_csv(filepath, encoding)
        elif ext in (".xlsx", ".xls"):
            df = _load_excel(filepath)
        elif ext == ".json":
            df = pl.read_json(filepath)
        elif ext == ".parquet":
            df = pl.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported: {ext}")
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        raise

    _validate(df)
    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def _load_csv(filepath: str, encoding: Optional[str] = None) -> pl.DataFrame:
    encodings_to_try = [encoding] if encoding else ["utf-8", "latin-1", "cp1252"]

    for enc in encodings_to_try:
        try:
            return pl.read_csv(filepath, encoding=enc, infer_schema_length=10000)
        except Exception:
            logger.debug(f"Failed with encoding {enc}, trying next...")
            continue

    raise ValueError(f"Could not read CSV with any encoding: {encodings_to_try}")


def _load_excel(filepath: str) -> pl.DataFrame:
    import pandas as pd
    pdf = pd.read_excel(filepath, engine="openpyxl")
    return pl.from_pandas(pdf)


def _validate(df: pl.DataFrame):
    if df.shape[0] == 0:
        raise ValueError("Dataset is empty (0 rows)")
    if df.shape[1] == 0:
        raise ValueError("Dataset has no columns")


def get_metadata(filepath: str, df: pl.DataFrame) -> dict:
    path = Path(filepath)
    return {
        "filename": path.name,
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns,
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
    }
