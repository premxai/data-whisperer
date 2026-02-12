import polars as pl
import pytest
from unittest.mock import patch, MagicMock

from app.agents import detective, statistician, visualizer, storyteller
from app.analyzers.profiler import profile_dataset
from app.analyzers.outliers import detect_outliers
from app.analyzers.correlations import analyze_correlations
from app.analyzers.timeseries import analyze_timeseries


@pytest.fixture
def sample_df():
    return pl.DataFrame({
        "id": list(range(1, 101)),
        "age": [25 + (i % 40) for i in range(100)],
        "salary": [40000 + i * 500 for i in range(100)],
        "department": ["Engineering", "Marketing", "Sales", "HR"] * 25,
        "rating": [3.0 + (i % 5) * 0.5 for i in range(100)],
    })


@pytest.fixture
def profile(sample_df):
    return profile_dataset(sample_df)


@pytest.fixture
def outliers(sample_df, profile):
    return detect_outliers(sample_df, profile["columns"])


@pytest.fixture
def correlations(sample_df, profile):
    return analyze_correlations(sample_df, profile["columns"])


@pytest.fixture
def timeseries_result(sample_df, profile):
    return analyze_timeseries(sample_df, profile["columns"])


class TestDetectiveAgent:
    def test_fallback_runs_without_llm(self, profile, outliers):
        """Detective should work even when Ollama is down."""
        with patch("app.agents.detective.generate_json", side_effect=ConnectionError("no ollama")):
            result = detective.run(profile, outliers)

        assert "findings" in result
        assert "overall_quality_score" in result
        assert isinstance(result["findings"], list)
        assert 0 <= result["overall_quality_score"] <= 100

    def test_fallback_detects_nulls(self):
        df = pl.DataFrame({
            "a": [1, 2, None, None, None, None, None, None, None, None],
            "b": list(range(10)),
        })
        profile = profile_dataset(df)
        outliers = detect_outliers(df, profile["columns"])

        result = detective._fallback_analysis(profile, outliers)
        titles = [f["title"] for f in result["findings"]]
        assert any("missing" in t.lower() for t in titles)

    def test_fallback_detects_duplicates(self):
        df = pl.DataFrame({"a": [1, 1, 2, 2], "b": ["x", "x", "y", "y"]})
        profile = profile_dataset(df)
        outliers = detect_outliers(df, profile["columns"])

        result = detective._fallback_analysis(profile, outliers)
        titles = [f["title"] for f in result["findings"]]
        assert any("duplicate" in t.lower() for t in titles)

    def test_build_context_format(self, profile, outliers):
        ctx = detective.build_context(profile, outliers)
        assert isinstance(ctx, str)
        assert "rows" in ctx.lower() or "100" in ctx


class TestStatisticianAgent:
    def test_normality_tests(self, sample_df, profile):
        numeric_cols = [c for c, i in profile["columns"].items() if i["detected_type"] == "numeric"]
        results = statistician.normality_tests(sample_df, numeric_cols)
        assert isinstance(results, list)
        for r in results:
            assert "column" in r
            assert "p_value" in r
            assert "is_normal" in r

    def test_chi_square_tests(self, sample_df, profile):
        cat_cols = [c for c, i in profile["columns"].items() if i["detected_type"] == "categorical"]
        results = statistician.chi_square_tests(sample_df, cat_cols)
        assert isinstance(results, list)

    def test_fallback_interpretation(self):
        raw = {
            "normality_tests": [
                {"column": "age", "p_value": 0.03, "is_normal": False, "skewness": 0.5}
            ],
            "chi_square_tests": [
                {"columns": ["a", "b"], "significant": True, "p_value": 0.01}
            ],
            "high_correlations": [],
        }
        result = statistician._fallback_interpretation(raw)
        assert "interpretations" in result
        assert "key_takeaways" in result
        assert len(result["interpretations"]) > 0

    def test_full_run_without_llm(self, sample_df, profile, correlations):
        with patch("app.agents.statistician.generate_json", side_effect=ConnectionError):
            result = statistician.run(sample_df, profile, correlations)

        assert "raw_results" in result
        assert "interpretation" in result


class TestVisualizerAgent:
    def test_fallback_chart_plan(self, profile, correlations, timeseries_result):
        charts = visualizer._fallback_chart_plan(profile, correlations, timeseries_result)
        assert isinstance(charts, list)
        assert len(charts) > 0
        for c in charts:
            assert "chart_type" in c
            assert "columns" in c
            assert "title" in c

    def test_create_histogram(self, sample_df):
        fig = visualizer.create_histogram(sample_df, "age", "Age Distribution")
        assert fig is not None
        assert hasattr(fig, "to_html")

    def test_create_box(self, sample_df):
        fig = visualizer.create_box(sample_df, ["age", "salary"], "Box Plots")
        assert fig is not None

    def test_create_bar(self, sample_df):
        fig = visualizer.create_bar(sample_df, "department", "Department Counts")
        assert fig is not None

    def test_create_scatter(self, sample_df):
        fig = visualizer.create_scatter(sample_df, "age", "salary", "Age vs Salary")
        assert fig is not None

    def test_full_run_without_llm(self, sample_df, profile, correlations, timeseries_result):
        with patch("app.agents.visualizer.generate_json", side_effect=ConnectionError):
            result = visualizer.run(sample_df, profile, correlations, timeseries_result)

        assert isinstance(result, list)
        assert len(result) > 0
        for v in result:
            assert "type" in v
            assert "title" in v
            assert "figure" in v


class TestStorytellerAgent:
    def test_fallback_report(self, profile, correlations, timeseries_result):
        detective_results = {"findings": [], "overall_quality_score": 85}
        stats_results = {
            "interpretation": {"key_takeaways": ["Test takeaway"], "interpretations": []}
        }

        report = storyteller._fallback_report(profile, detective_results,
                                               stats_results, timeseries_result, correlations)
        assert isinstance(report, str)
        assert "EXECUTIVE SUMMARY" in report
        assert len(report) > 100

    def test_split_sections(self):
        text = (
            "Executive Summary\nThis is the summary.\n\n"
            "Data Quality\nQuality is good.\n\n"
            "Key Findings\nFound some things.\n\n"
            "Recommendations\nDo this and that."
        )
        sections = storyteller._split_sections(text)
        assert "executive_summary" in sections
        assert "data_quality" in sections
        assert "recommendations" in sections

    def test_build_report_context(self, profile, correlations, timeseries_result):
        detective_results = {"findings": [{"severity": "high", "title": "test", "description": "desc"}],
                            "overall_quality_score": 75}
        stats_results = {"interpretation": {"key_takeaways": ["t1"], "interpretations": []}}

        ctx = storyteller.build_report_context(profile, detective_results, stats_results,
                                                timeseries_result, correlations)
        assert isinstance(ctx, str)
        assert "DATASET OVERVIEW" in ctx

    def test_full_run_without_llm(self, profile, correlations, timeseries_result):
        detective_results = {"findings": [], "overall_quality_score": 90}
        stats_results = {"interpretation": {"key_takeaways": [], "interpretations": []}}

        with patch("app.agents.storyteller.generate", side_effect=ConnectionError):
            result = storyteller.run(profile, detective_results, stats_results,
                                     timeseries_result, correlations)

        assert "report_text" in result
        assert "quality_score" in result
        assert len(result["report_text"]) > 0


class TestOrchestrator:
    def test_workflow_builds(self):
        from app.agents.orchestrator import build_workflow
        workflow = build_workflow()
        assert workflow is not None

    def test_full_pipeline_without_llm(self, sample_df):
        """Integration test: run the full pipeline with mocked LLM."""
        from app.agents.orchestrator import run_analysis

        with patch("app.agents.detective.generate_json", side_effect=ConnectionError):
            with patch("app.agents.statistician.generate_json", side_effect=ConnectionError):
                with patch("app.agents.visualizer.generate_json", side_effect=ConnectionError):
                    with patch("app.agents.storyteller.generate", side_effect=ConnectionError):
                        with patch("app.utils.embeddings.embed_dataset_schema"):
                            result = run_analysis("test-123", sample_df, {"filename": "test.csv"})

        assert result["status"] == "complete"
        assert "profile" in result
        assert "detective_results" in result
        assert "stats_results" in result
        assert "visualizations" in result
        assert "report" in result
