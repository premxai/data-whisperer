import os
import tempfile

import polars as pl
import pytest

from app.utils.data_loader import load_file, get_metadata, detect_file_type
from app.analyzers.profiler import (
    profile_dataset, profile_column, detect_column_type,
    profile_numeric, profile_categorical, detect_pii,
)


@pytest.fixture
def sample_csv(tmp_path):
    csv_content = (
        "id,name,age,salary,department,email,hire_date\n"
        "1,Alice,30,75000,Engineering,alice@example.com,2020-01-15\n"
        "2,Bob,25,65000,Marketing,bob@test.com,2021-03-20\n"
        "3,Carol,,80000,Engineering,carol@example.com,2019-06-10\n"
        "4,Dave,35,70000,Sales,,2022-01-05\n"
        "5,Eve,28,72000,Marketing,eve@test.com,2020-11-30\n"
        "6,Frank,45,95000,Engineering,frank@example.com,2018-04-22\n"
        "7,Grace,32,68000,Sales,grace@test.com,2021-08-14\n"
        "8,Hank,29,,Marketing,hank@example.com,2022-05-01\n"
        "9,Iris,40,88000,Engineering,iris@test.com,2017-09-18\n"
        "10,Jack,27,63000,Sales,jack@example.com,2023-01-10\n"
    )
    filepath = tmp_path / "test_data.csv"
    filepath.write_text(csv_content)
    return str(filepath)


@pytest.fixture
def sample_df(sample_csv):
    return load_file(sample_csv)


class TestDataLoader:
    def test_detect_csv(self):
        assert detect_file_type("data.csv") == ".csv"

    def test_detect_excel(self):
        assert detect_file_type("data.xlsx") == ".xlsx"

    def test_detect_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported"):
            detect_file_type("data.txt")

    def test_load_csv(self, sample_csv):
        df = load_file(sample_csv)
        assert df.shape[0] == 10
        assert df.shape[1] == 7

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_file("/nonexistent/file.csv")

    def test_metadata(self, sample_csv, sample_df):
        meta = get_metadata(sample_csv, sample_df)
        assert meta["rows"] == 10
        assert meta["columns"] == 7
        assert "id" in meta["column_names"]
        assert meta["file_size_mb"] > 0


class TestProfiler:
    def test_profile_dataset(self, sample_df):
        profile = profile_dataset(sample_df)
        assert "shape" in profile
        assert profile["shape"]["rows"] == 10
        assert profile["shape"]["columns"] == 7
        assert "columns" in profile
        assert "completeness_pct" in profile

    def test_detects_numeric(self):
        s = pl.Series("nums", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert detect_column_type(s) == "numeric"

    def test_detects_categorical(self):
        s = pl.Series("cats", ["a", "b", "a", "b", "a"] * 100)
        assert detect_column_type(s) == "categorical"

    def test_detects_text(self):
        s = pl.Series("text", [f"unique_string_{i}" for i in range(100)])
        assert detect_column_type(s) == "text"

    def test_profile_numeric_stats(self):
        s = pl.Series("nums", [10, 20, 30, 40, 50])
        stats = profile_numeric(s)
        assert stats["mean"] == 30.0
        assert stats["min"] == 10
        assert stats["max"] == 50

    def test_profile_numeric_with_nulls(self):
        s = pl.Series("nums", [10, None, 30, None, 50])
        stats = profile_numeric(s)
        assert stats["mean"] == 30.0

    def test_profile_categorical_counts(self):
        s = pl.Series("cats", ["a", "b", "a", "c", "a", "b"])
        stats = profile_categorical(s)
        assert stats["n_unique"] == 3
        assert len(stats["top_values"]) > 0

    def test_null_tracking(self, sample_df):
        profile = profile_dataset(sample_df)
        cols = profile["columns"]
        # age has 1 null, salary has 1 null, email has 1 null
        age_nulls = cols["age"]["null_count"]
        assert age_nulls >= 1

    def test_pii_detection(self):
        s = pl.Series("emails", [
            "alice@example.com", "bob@test.com", "carol@example.com",
            "dave@mail.com", "eve@test.com",
        ])
        flags = detect_pii(s)
        assert "potential_email" in flags

    def test_pii_no_false_positive(self):
        s = pl.Series("names", ["Alice", "Bob", "Carol", "Dave", "Eve"])
        flags = detect_pii(s)
        assert "potential_email" not in flags

    def test_completeness_calculation(self, sample_df):
        profile = profile_dataset(sample_df)
        assert 0 < profile["completeness_pct"] <= 100

    def test_duplicate_detection(self):
        df = pl.DataFrame({
            "a": [1, 2, 1, 2],
            "b": ["x", "y", "x", "y"],
        })
        profile = profile_dataset(df)
        assert profile["duplicate_rows"] == 2

    def test_type_counts(self, sample_df):
        profile = profile_dataset(sample_df)
        types = profile["type_counts"]
        total = sum(types.values())
        assert total == sample_df.shape[1]
