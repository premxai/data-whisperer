import logging
from typing import Any

from app.utils.llm_client import generate_json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a data quality detective. Given data profiling results and outlier analysis,
identify potential data quality issues, anomalies, and suspicious patterns.

Be specific and actionable. Rate each finding by severity (high/medium/low).
Respond in JSON with this structure:
{
    "findings": [
        {
            "title": "short description",
            "severity": "high|medium|low",
            "description": "detailed explanation",
            "affected_columns": ["col1"],
            "recommendation": "what to do about it"
        }
    ],
    "overall_quality_score": 0-100
}"""


def build_context(profile: dict, outliers: dict) -> str:
    """Summarize profile and outlier data for the LLM."""
    lines = []

    shape = profile.get("shape", {})
    lines.append(f"Dataset: {shape.get('rows', '?')} rows, {shape.get('columns', '?')} columns")
    lines.append(f"Completeness: {profile.get('completeness_pct', '?')}%")
    lines.append(f"Duplicate rows: {profile.get('duplicate_rows', 0)}")
    lines.append("")

    for col_name, col_info in profile.get("columns", {}).items():
        line = f"- {col_name} ({col_info['detected_type']}): "
        line += f"{col_info['null_pct']}% null, {col_info['n_unique']} unique"

        if col_info.get("pii_flags"):
            line += f" [PII: {', '.join(col_info['pii_flags'])}]"

        stats = col_info.get("stats", {})
        if "mean" in stats:
            line += f", mean={stats['mean']}, std={stats.get('std')}"

        lines.append(line)

    # outlier summary
    univariate = outliers.get("univariate", {})
    if univariate:
        lines.append("\nOutliers (IQR method):")
        for col, info in univariate.items():
            lines.append(f"- {col}: {info['count']} outliers ({info['pct']}%)")

    multi = outliers.get("multivariate", {})
    if multi.get("count", 0) > 0:
        lines.append(f"\nMultivariate outliers (Isolation Forest): {multi['count']} ({multi['pct']}%)")

    return "\n".join(lines)


def run(profile: dict, outliers: dict) -> dict:
    """Run the detective agent to find data quality issues."""
    logger.info("Detective agent starting analysis")

    context = build_context(profile, outliers)
    prompt = f"Analyze this dataset for quality issues and anomalies:\n\n{context}"

    try:
        result = generate_json(prompt, system=SYSTEM_PROMPT, max_tokens=2048)

        if result.get("parse_error"):
            logger.warning("Detective got unparseable LLM response, using fallback")
            result = _fallback_analysis(profile, outliers)

    except ConnectionError:
        logger.error("LLM unavailable, using rule-based fallback")
        result = _fallback_analysis(profile, outliers)

    n_findings = len(result.get("findings", []))
    logger.info(f"Detective found {n_findings} issues")
    return result


def _fallback_analysis(profile: dict, outliers: dict) -> dict:
    """Rule-based fallback when LLM is unavailable."""
    findings = []

    # check for high null percentages
    for col_name, col_info in profile.get("columns", {}).items():
        if col_info["null_pct"] > 50:
            findings.append({
                "title": f"High missing rate in '{col_name}'",
                "severity": "high",
                "description": f"{col_info['null_pct']}% of values are missing.",
                "affected_columns": [col_name],
                "recommendation": "Consider dropping or imputing this column.",
            })
        elif col_info["null_pct"] > 10:
            findings.append({
                "title": f"Notable missing values in '{col_name}'",
                "severity": "medium",
                "description": f"{col_info['null_pct']}% missing.",
                "affected_columns": [col_name],
                "recommendation": "Investigate why data is missing. Impute if appropriate.",
            })

        if col_info.get("pii_flags"):
            findings.append({
                "title": f"Potential PII in '{col_name}'",
                "severity": "high",
                "description": f"Detected patterns: {', '.join(col_info['pii_flags'])}",
                "affected_columns": [col_name],
                "recommendation": "Review and mask/remove PII before sharing.",
            })

    # check duplicates
    dupes = profile.get("duplicate_rows", 0)
    if dupes > 0:
        findings.append({
            "title": "Duplicate rows detected",
            "severity": "medium" if dupes < 100 else "high",
            "description": f"{dupes} duplicate rows found.",
            "affected_columns": [],
            "recommendation": "Review and deduplicate if not intentional.",
        })

    # check outliers
    for col, info in outliers.get("univariate", {}).items():
        if info["pct"] > 5:
            findings.append({
                "title": f"High outlier rate in '{col}'",
                "severity": "medium",
                "description": f"{info['count']} outliers ({info['pct']}%) outside IQR bounds.",
                "affected_columns": [col],
                "recommendation": "Check if these are data entry errors or genuine extreme values.",
            })

    # calculate quality score
    completeness = profile.get("completeness_pct", 100)
    dupe_penalty = min(dupes / max(profile["shape"]["rows"], 1) * 100, 20)
    outlier_penalty = min(len(outliers.get("univariate", {})) * 3, 15)
    score = max(0, completeness - dupe_penalty - outlier_penalty)

    return {
        "findings": findings,
        "overall_quality_score": round(score),
    }
