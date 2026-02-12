import logging
from typing import Any

from app.utils.llm_client import generate

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a data analysis report writer. Given analysis results from multiple agents,
write a clear, professional report in plain English. No jargon without explanation.

Structure your report with these sections:
1. Executive Summary (2-3 sentences overview)
2. Data Quality Assessment (issues found, completeness, recommendations)
3. Key Statistical Findings (what the numbers tell us)
4. Patterns and Relationships (correlations, trends, clusters)
5. Recommendations (actionable next steps)

Write in a direct, informative tone. Like a senior analyst writing for a mixed audience.
Do not use markdown headers - use plain text section titles followed by content.
Keep it under 1500 words."""


def build_report_context(profile: dict, detective_results: dict,
                          stats_results: dict, timeseries: dict,
                          correlations: dict) -> str:
    """Assemble all analysis outputs into a context string for the LLM."""
    sections = []

    # dataset overview
    shape = profile.get("shape", {})
    sections.append(
        f"DATASET OVERVIEW\n"
        f"Rows: {shape.get('rows')}, Columns: {shape.get('columns')}\n"
        f"Completeness: {profile.get('completeness_pct')}%\n"
        f"Duplicate rows: {profile.get('duplicate_rows')}\n"
        f"Type breakdown: {profile.get('type_counts')}"
    )

    # detective findings
    findings = detective_results.get("findings", [])
    if findings:
        finding_lines = []
        for f in findings:
            finding_lines.append(f"- [{f['severity'].upper()}] {f['title']}: {f['description']}")
        sections.append("DATA QUALITY FINDINGS\n" + "\n".join(finding_lines))
        sections.append(f"Quality score: {detective_results.get('overall_quality_score', 'N/A')}/100")

    # statistical interpretations
    interp = stats_results.get("interpretation", {})
    takeaways = interp.get("key_takeaways", [])
    if takeaways:
        sections.append("STATISTICAL TAKEAWAYS\n" + "\n".join(f"- {t}" for t in takeaways))

    stat_interps = interp.get("interpretations", [])
    if stat_interps:
        lines = []
        for si in stat_interps[:10]:
            lines.append(f"- {si.get('test_name', 'Test')} on {si.get('columns', [])}: {si.get('result_summary', '')}")
        sections.append("STATISTICAL DETAILS\n" + "\n".join(lines))

    # correlations
    high_corr = correlations.get("numeric", {}).get("high_correlations", [])
    if high_corr:
        lines = [f"- {c['col1']} <-> {c['col2']}: r={c['correlation']}" for c in high_corr[:5]]
        sections.append("HIGH CORRELATIONS\n" + "\n".join(lines))

    # time series
    if timeseries.get("has_time_component"):
        analyses = timeseries.get("analyses", {})
        if analyses:
            lines = []
            for col, info in list(analyses.items())[:5]:
                trend = info.get("trend", {})
                season = info.get("seasonality", {})
                lines.append(f"- {col}: trend={trend.get('direction', 'unknown')}, "
                           f"seasonal={'yes' if season.get('detected') else 'no'}")
            sections.append("TIME SERIES PATTERNS\n" + "\n".join(lines))

    return "\n\n".join(sections)


def run(profile: dict, detective_results: dict, stats_results: dict,
        timeseries: dict, correlations: dict) -> dict:
    """Generate the final narrative report."""
    logger.info("Storyteller agent writing report")

    context = build_report_context(profile, detective_results, stats_results,
                                    timeseries, correlations)

    prompt = (
        f"Write a data analysis report based on these findings:\n\n{context}\n\n"
        f"Remember: Executive Summary, Data Quality, Key Findings, Patterns, Recommendations."
    )

    try:
        report_text = generate(prompt, system=SYSTEM_PROMPT, max_tokens=3000, temperature=0.4)
    except ConnectionError:
        logger.warning("LLM unavailable for storyteller, using template fallback")
        report_text = _fallback_report(profile, detective_results, stats_results,
                                        timeseries, correlations)

    logger.info(f"Report generated ({len(report_text)} chars)")

    return {
        "report_text": report_text,
        "quality_score": detective_results.get("overall_quality_score", 0),
        "n_findings": len(detective_results.get("findings", [])),
        "sections": _split_sections(report_text),
    }


def _split_sections(text: str) -> dict:
    """Try to split report into named sections."""
    sections = {}
    current_section = "introduction"
    current_lines = []

    keywords = {
        "executive summary": "executive_summary",
        "data quality": "data_quality",
        "key findings": "key_findings",
        "key statistical": "key_findings",
        "patterns": "patterns",
        "relationships": "patterns",
        "recommendations": "recommendations",
    }

    for line in text.split("\n"):
        lower = line.strip().lower()
        matched = False
        for keyword, section_name in keywords.items():
            if keyword in lower and len(lower) < 50:
                if current_lines:
                    sections[current_section] = "\n".join(current_lines).strip()
                current_section = section_name
                current_lines = []
                matched = True
                break
        if not matched:
            current_lines.append(line)

    if current_lines:
        sections[current_section] = "\n".join(current_lines).strip()

    return sections


def _fallback_report(profile, detective, stats, timeseries, correlations) -> str:
    """Template-based report when LLM is unavailable."""
    shape = profile.get("shape", {})
    quality_score = detective.get("overall_quality_score", 0)
    findings = detective.get("findings", [])

    lines = [
        "EXECUTIVE SUMMARY",
        "",
        f"This dataset contains {shape.get('rows', 0)} rows and {shape.get('columns', 0)} columns. "
        f"Overall data quality score: {quality_score}/100. "
        f"The analysis identified {len(findings)} potential issues.",
        "",
        "DATA QUALITY ASSESSMENT",
        "",
        f"Completeness: {profile.get('completeness_pct', 0)}% of cells contain values.",
        f"Duplicate rows: {profile.get('duplicate_rows', 0)}.",
        "",
    ]

    if findings:
        for f in findings:
            lines.append(f"[{f['severity'].upper()}] {f['title']}")
            lines.append(f"  {f['description']}")
            lines.append(f"  Recommendation: {f['recommendation']}")
            lines.append("")

    lines.append("KEY FINDINGS")
    lines.append("")

    takeaways = stats.get("interpretation", {}).get("key_takeaways", [])
    for t in takeaways:
        lines.append(f"- {t}")

    high_corr = correlations.get("numeric", {}).get("high_correlations", [])
    if high_corr:
        lines.append("")
        lines.append("PATTERNS AND RELATIONSHIPS")
        lines.append("")
        for c in high_corr[:5]:
            lines.append(f"- Strong correlation between {c['col1']} and {c['col2']} (r={c['correlation']})")

    lines.append("")
    lines.append("RECOMMENDATIONS")
    lines.append("")
    lines.append("- Address any high-severity data quality issues before further analysis.")
    lines.append("- Investigate highly correlated variables for potential redundancy.")
    if profile.get("duplicate_rows", 0) > 0:
        lines.append("- Review and deduplicate rows if appropriate.")

    return "\n".join(lines)
