import logging
import time
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

import polars as pl

from app.analyzers.profiler import profile_dataset
from app.analyzers.outliers import detect_outliers
from app.analyzers.correlations import analyze_correlations
from app.analyzers.timeseries import analyze_timeseries
from app.agents import detective, statistician, visualizer, storyteller
from app.utils.embeddings import embed_dataset_schema

logger = logging.getLogger(__name__)


class AnalysisState(TypedDict, total=False):
    upload_id: str
    df: Any  # polars DataFrame
    metadata: dict
    profile: dict
    outliers: dict
    correlations: dict
    timeseries: dict
    detective_results: dict
    stats_results: dict
    visualizations: list
    report: dict
    status: str
    error: str
    start_time: float


def profiling_node(state: AnalysisState) -> dict:
    """Profile the dataset and detect outliers."""
    logger.info("Running profiling node")
    df = state["df"]

    profile = profile_dataset(df)
    column_types = profile["columns"]

    outliers = detect_outliers(df, column_types)
    correlations = analyze_correlations(df, column_types)
    timeseries = analyze_timeseries(df, column_types)

    # embed schema for RAG
    try:
        sample_rows = df.head(10).to_dicts()
        embed_dataset_schema(state["upload_id"], state.get("metadata", {}),
                            profile, sample_rows)
    except Exception as e:
        logger.warning(f"Failed to embed schema: {e}")

    return {
        "profile": profile,
        "outliers": outliers,
        "correlations": correlations,
        "timeseries": timeseries,
        "status": "profiling_complete",
    }


def detective_node(state: AnalysisState) -> dict:
    """Run the detective agent for anomaly detection."""
    logger.info("Running detective node")
    results = detective.run(state["profile"], state["outliers"])
    return {"detective_results": results, "status": "detective_complete"}


def statistician_node(state: AnalysisState) -> dict:
    """Run the statistician agent."""
    logger.info("Running statistician node")
    results = statistician.run(state["df"], state["profile"], state["correlations"])
    return {"stats_results": results, "status": "statistician_complete"}


def visualizer_node(state: AnalysisState) -> dict:
    """Run the visualizer agent."""
    logger.info("Running visualizer node")
    charts = visualizer.run(state["df"], state["profile"],
                           state["correlations"], state["timeseries"])
    return {"visualizations": charts, "status": "visualizer_complete"}


def storyteller_node(state: AnalysisState) -> dict:
    """Run the storyteller agent to generate the report."""
    logger.info("Running storyteller node")
    report = storyteller.run(
        state["profile"],
        state["detective_results"],
        state["stats_results"],
        state["timeseries"],
        state["correlations"],
    )
    elapsed = time.time() - state.get("start_time", time.time())
    report["elapsed_seconds"] = round(elapsed, 1)
    return {"report": report, "status": "complete"}


def build_workflow() -> StateGraph:
    """Construct the LangGraph analysis workflow."""
    workflow = StateGraph(AnalysisState)

    workflow.add_node("profiling", profiling_node)
    workflow.add_node("detective", detective_node)
    workflow.add_node("statistician", statistician_node)
    workflow.add_node("visualizer", visualizer_node)
    workflow.add_node("storyteller", storyteller_node)

    # profiling must run first
    workflow.set_entry_point("profiling")

    # after profiling, detective and statistician can run
    # (LangGraph doesn't natively do parallel, so we chain them)
    workflow.add_edge("profiling", "detective")
    workflow.add_edge("detective", "statistician")
    workflow.add_edge("statistician", "visualizer")
    workflow.add_edge("visualizer", "storyteller")
    workflow.add_edge("storyteller", END)

    return workflow


def run_analysis(upload_id: str, df: pl.DataFrame, metadata: dict) -> dict:
    """Execute the full analysis pipeline."""
    logger.info(f"Starting analysis for upload {upload_id}")

    initial_state = {
        "upload_id": upload_id,
        "df": df,
        "metadata": metadata,
        "status": "started",
        "start_time": time.time(),
    }

    workflow = build_workflow()
    app = workflow.compile()

    try:
        final_state = app.invoke(initial_state)
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "upload_id": upload_id,
        }

    # strip the dataframe from results (not serializable)
    result = {k: v for k, v in final_state.items() if k != "df"}
    result["status"] = "complete"

    elapsed = time.time() - initial_state["start_time"]
    logger.info(f"Analysis complete for {upload_id} in {elapsed:.1f}s")

    return result
