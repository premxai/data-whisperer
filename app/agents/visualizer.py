import logging
from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from app.utils.llm_client import generate_json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a data visualization expert. Given a dataset profile and analysis results,
decide which visualizations would be most informative.

Respond in JSON:
{
    "charts": [
        {
            "chart_type": "histogram|scatter|bar|heatmap|box|line|pie",
            "columns": ["col1", "col2"],
            "title": "descriptive title",
            "reason": "why this chart is useful"
        }
    ]
}

Pick 5-8 charts that tell the most interesting story about the data. Prioritize:
- Distribution of key numeric variables
- Relationships between highly correlated columns
- Category breakdowns
- Time trends if applicable
- Outlier visualization"""

MAX_CHARTS = 10


def plan_charts(profile: dict, correlations: dict, timeseries: dict) -> list[dict]:
    """Use LLM to decide which charts to create, with rule-based fallback."""
    context = _build_context(profile, correlations, timeseries)
    prompt = f"What visualizations should I create for this dataset?\n\n{context}"

    try:
        result = generate_json(prompt, system=SYSTEM_PROMPT)
        if not result.get("parse_error") and "charts" in result:
            return result["charts"][:MAX_CHARTS]
    except Exception as e:
        logger.warning(f"LLM chart planning failed: {e}")

    return _fallback_chart_plan(profile, correlations, timeseries)


def _build_context(profile: dict, correlations: dict, timeseries: dict) -> str:
    lines = []
    cols = profile.get("columns", {})

    lines.append("Columns:")
    for name, info in cols.items():
        lines.append(f"  {name}: {info['detected_type']}, {info['n_unique']} unique, {info['null_pct']}% null")

    high_corr = correlations.get("numeric", {}).get("high_correlations", [])
    if high_corr:
        lines.append("\nHigh correlations:")
        for c in high_corr[:5]:
            lines.append(f"  {c['col1']} <-> {c['col2']}: {c['correlation']}")

    if timeseries.get("has_time_component"):
        lines.append(f"\nTime columns: {timeseries.get('time_columns', [])}")

    return "\n".join(lines)


def _fallback_chart_plan(profile: dict, correlations: dict, timeseries: dict) -> list[dict]:
    """Rule-based chart selection when LLM is unavailable."""
    charts = []
    cols = profile.get("columns", {})

    numeric_cols = [c for c, i in cols.items() if i["detected_type"] == "numeric"]
    categorical_cols = [c for c, i in cols.items() if i["detected_type"] == "categorical"]

    # histograms for numeric
    for col in numeric_cols[:3]:
        charts.append({
            "chart_type": "histogram",
            "columns": [col],
            "title": f"Distribution of {col}",
            "reason": "Shows the spread and shape of values",
        })

    # box plots for numeric
    if numeric_cols:
        charts.append({
            "chart_type": "box",
            "columns": numeric_cols[:6],
            "title": "Box plots of numeric columns",
            "reason": "Compare distributions and spot outliers",
        })

    # bar charts for categorical
    for col in categorical_cols[:2]:
        charts.append({
            "chart_type": "bar",
            "columns": [col],
            "title": f"Value counts for {col}",
            "reason": "Shows category distribution",
        })

    # scatter for top correlation
    high_corr = correlations.get("numeric", {}).get("high_correlations", [])
    for c in high_corr[:2]:
        charts.append({
            "chart_type": "scatter",
            "columns": [c["col1"], c["col2"]],
            "title": f"{c['col1']} vs {c['col2']} (r={c['correlation']})",
            "reason": f"High correlation: {c['correlation']}",
        })

    # correlation heatmap
    if len(numeric_cols) >= 3:
        charts.append({
            "chart_type": "heatmap",
            "columns": numeric_cols[:10],
            "title": "Correlation heatmap",
            "reason": "Overview of numeric relationships",
        })

    # time series line
    if timeseries.get("has_time_component"):
        time_col = timeseries.get("time_columns", [None])[0]
        for col in numeric_cols[:2]:
            charts.append({
                "chart_type": "line",
                "columns": [time_col, col],
                "title": f"{col} over time",
                "reason": "Time trend visualization",
            })

    return charts[:MAX_CHARTS]


def create_histogram(df: pl.DataFrame, col: str, title: str) -> go.Figure:
    values = df[col].drop_nulls().to_list()
    fig = px.histogram(x=values, nbins=50, title=title, labels={"x": col, "y": "Count"})
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def create_box(df: pl.DataFrame, cols: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    for col in cols:
        values = df[col].drop_nulls().to_list()
        fig.add_trace(go.Box(y=values, name=col))
    fig.update_layout(title=title, template="plotly_white")
    return fig


def create_bar(df: pl.DataFrame, col: str, title: str) -> go.Figure:
    counts = df[col].value_counts().sort("count", descending=True).head(20)
    fig = px.bar(
        x=[str(v) for v in counts[col].to_list()],
        y=counts["count"].to_list(),
        title=title,
        labels={"x": col, "y": "Count"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def create_scatter(df: pl.DataFrame, col_x: str, col_y: str, title: str) -> go.Figure:
    # sample if too many points
    sample = df.select([col_x, col_y]).drop_nulls()
    if len(sample) > 5000:
        sample = sample.sample(5000, seed=42)

    fig = px.scatter(
        x=sample[col_x].to_list(),
        y=sample[col_y].to_list(),
        title=title,
        labels={"x": col_x, "y": col_y},
        opacity=0.5,
    )
    fig.update_layout(template="plotly_white")
    return fig


def create_heatmap(df: pl.DataFrame, cols: list[str], title: str) -> go.Figure:
    subset = df.select(cols).fill_null(strategy="mean")
    try:
        matrix = subset.to_numpy().astype(np.float64)
        corr = np.corrcoef(matrix, rowvar=False)
    except Exception:
        return go.Figure()

    fig = go.Figure(data=go.Heatmap(
        z=corr, x=cols, y=cols,
        colorscale="RdBu_r", zmid=0,
        text=np.round(corr, 2), texttemplate="%{text}",
    ))
    fig.update_layout(title=title, template="plotly_white")
    return fig


def create_line(df: pl.DataFrame, time_col: str, value_col: str, title: str) -> go.Figure:
    sorted_df = df.select([time_col, value_col]).drop_nulls().sort(time_col)
    fig = px.line(
        x=sorted_df[time_col].to_list(),
        y=sorted_df[value_col].to_list(),
        title=title,
        labels={"x": time_col, "y": value_col},
    )
    fig.update_layout(template="plotly_white")
    return fig


def create_pie(df: pl.DataFrame, col: str, title: str) -> go.Figure:
    counts = df[col].value_counts().sort("count", descending=True).head(10)
    fig = px.pie(
        names=[str(v) for v in counts[col].to_list()],
        values=counts["count"].to_list(),
        title=title,
    )
    fig.update_layout(template="plotly_white")
    return fig


CHART_BUILDERS = {
    "histogram": lambda df, spec: create_histogram(df, spec["columns"][0], spec["title"]),
    "box": lambda df, spec: create_box(df, spec["columns"], spec["title"]),
    "bar": lambda df, spec: create_bar(df, spec["columns"][0], spec["title"]),
    "scatter": lambda df, spec: create_scatter(df, spec["columns"][0], spec["columns"][1], spec["title"]),
    "heatmap": lambda df, spec: create_heatmap(df, spec["columns"], spec["title"]),
    "line": lambda df, spec: create_line(df, spec["columns"][0], spec["columns"][1], spec["title"]),
    "pie": lambda df, spec: create_pie(df, spec["columns"][0], spec["title"]),
}


def run(df: pl.DataFrame, profile: dict, correlations: dict, timeseries: dict) -> list[dict]:
    """Generate all visualizations for the dataset."""
    logger.info("Visualizer agent starting")

    chart_plan = plan_charts(profile, correlations, timeseries)
    logger.info(f"Planning {len(chart_plan)} charts")

    results = []
    available_cols = set(df.columns)

    for spec in chart_plan:
        chart_type = spec.get("chart_type", "")
        cols = spec.get("columns", [])

        # validate columns exist
        if not all(c in available_cols for c in cols):
            logger.debug(f"Skipping chart '{spec.get('title')}': missing columns")
            continue

        builder = CHART_BUILDERS.get(chart_type)
        if not builder:
            logger.debug(f"Unknown chart type: {chart_type}")
            continue

        try:
            fig = builder(df, spec)
            results.append({
                "type": chart_type,
                "title": spec["title"],
                "reason": spec.get("reason", ""),
                "figure": fig,
            })
        except Exception as e:
            logger.warning(f"Failed to create {chart_type} chart: {e}")

    logger.info(f"Visualizer created {len(results)} charts")
    return results
