import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 1100px;
    margin: 0 auto;
    padding: 20px 40px;
    color: #333;
    background: #fafafa;
    line-height: 1.6;
}
h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
h2 { color: #2c3e50; margin-top: 2em; }
.summary-box {
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.quality-score {
    font-size: 48px;
    font-weight: bold;
    text-align: center;
    padding: 20px;
}
.quality-high { color: #27ae60; }
.quality-mid { color: #f39c12; }
.quality-low { color: #e74c3c; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 16px 0;
    background: white;
}
th { background: #3498db; color: white; padding: 10px 12px; text-align: left; }
td { padding: 8px 12px; border-bottom: 1px solid #eee; }
tr:hover { background: #f5f5f5; }
.finding {
    padding: 12px 16px;
    margin: 8px 0;
    border-left: 4px solid;
    background: white;
    border-radius: 0 4px 4px 0;
}
.finding-high { border-color: #e74c3c; }
.finding-medium { border-color: #f39c12; }
.finding-low { border-color: #27ae60; }
.severity-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: bold;
    color: white;
}
.sev-high { background: #e74c3c; }
.sev-medium { background: #f39c12; }
.sev-low { background: #27ae60; }
.chart-container { margin: 24px 0; }
.collapsible {
    cursor: pointer;
    padding: 12px 16px;
    background: #ecf0f1;
    border: none;
    width: 100%;
    text-align: left;
    font-size: 16px;
    font-weight: bold;
    border-radius: 4px;
    margin-top: 16px;
}
.collapsible:hover { background: #d5dbdb; }
.collapsible-content { padding: 0 16px; display: none; overflow: hidden; }
.collapsible-content.active { display: block; }
.report-text { white-space: pre-wrap; background: white; padding: 20px; border-radius: 8px; }
"""

JS = """
document.querySelectorAll('.collapsible').forEach(function(btn) {
    btn.addEventListener('click', function() {
        var content = this.nextElementSibling;
        content.classList.toggle('active');
        this.textContent = content.classList.contains('active')
            ? this.textContent.replace('[+]', '[-]')
            : this.textContent.replace('[-]', '[+]');
    });
});
"""


def generate_html(upload_id: str, results: dict, output_dir: str) -> str:
    """Generate a standalone HTML report with interactive charts."""
    logger.info(f"Generating HTML report for {upload_id}")

    filepath = os.path.join(output_dir, f"report_{upload_id}.html")

    metadata = results.get("metadata", {})
    profile = results.get("profile", {})
    detective = results.get("detective_results", {})
    report = results.get("report", {})
    visualizations = results.get("visualizations", [])

    quality_score = report.get("quality_score", detective.get("overall_quality_score", 0))

    parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        f"<title>DataWhisperer Report - {metadata.get('filename', upload_id)}</title>",
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
        f"<style>{CSS}</style>",
        "</head>",
        "<body>",
        f"<h1>DataWhisperer Report</h1>",
        f"<p>Dataset: <strong>{metadata.get('filename', 'Unknown')}</strong> | "
        f"{metadata.get('rows', '?')} rows x {metadata.get('columns', '?')} columns | "
        f"{metadata.get('file_size_mb', '?')} MB</p>",
    ]

    # quality score
    score_class = "quality-high" if quality_score >= 70 else ("quality-mid" if quality_score >= 40 else "quality-low")
    parts.append(f'<div class="summary-box"><div class="quality-score {score_class}">{quality_score}/100</div>')
    parts.append(f'<p style="text-align:center">Data Quality Score</p></div>')

    # profile section
    parts.append(_build_profile_section(profile))

    # findings section
    findings = detective.get("findings", [])
    if findings:
        parts.append(_build_findings_section(findings))

    # visualizations
    if visualizations:
        parts.append('<h2>Visualizations</h2>')
        for i, viz in enumerate(visualizations):
            fig = viz.get("figure")
            if fig is None:
                continue
            try:
                chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
                parts.append(f'<div class="chart-container">')
                parts.append(chart_html)
                parts.append(f'</div>')
            except Exception as e:
                logger.debug(f"Could not embed chart: {e}")

    # report text
    report_text = report.get("report_text", "")
    if report_text:
        parts.append('<button class="collapsible">[+] Full Analysis Report</button>')
        parts.append(f'<div class="collapsible-content"><div class="report-text">{_escape(report_text)}</div></div>')

    # statistical details
    stats = results.get("stats_results", {})
    raw = stats.get("raw_results", {})
    if raw:
        parts.append('<button class="collapsible">[+] Statistical Details</button>')
        parts.append('<div class="collapsible-content">')
        parts.append(_build_stats_section(raw))
        parts.append('</div>')

    parts.append(f"<script>{JS}</script>")
    parts.append("</body></html>")

    html = "\n".join(parts)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"HTML report saved to {filepath}")
    return filepath


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")


def _build_profile_section(profile: dict) -> str:
    parts = ['<h2>Data Profile</h2>', '<div class="summary-box">']

    shape = profile.get("shape", {})
    parts.append(f'<p>Completeness: <strong>{profile.get("completeness_pct", "?")}%</strong> | ')
    parts.append(f'Duplicates: <strong>{profile.get("duplicate_rows", 0)}</strong> | ')
    parts.append(f'Total nulls: <strong>{profile.get("total_nulls", 0)}</strong></p>')

    types = profile.get("type_counts", {})
    parts.append(f'<p>Column types: {", ".join(f"{k}: {v}" for k, v in types.items())}</p>')

    # column table
    columns = profile.get("columns", {})
    if columns:
        parts.append('<table><tr><th>Column</th><th>Type</th><th>Null %</th><th>Unique</th><th>Details</th></tr>')
        for name, info in columns.items():
            stats = info.get("stats", {})
            detail = ""
            if "mean" in stats:
                detail = f"mean={stats['mean']}, std={stats.get('std', '')}"
            elif "top_values" in stats:
                top = stats["top_values"][:3]
                detail = ", ".join(f"{v['value']}({v['count']})" for v in top)

            pii = info.get("pii_flags", [])
            if pii:
                detail += f' <span style="color:red">[PII: {", ".join(pii)}]</span>'

            parts.append(f'<tr><td>{_escape(name)}</td><td>{info["detected_type"]}</td>'
                        f'<td>{info["null_pct"]}%</td><td>{info["n_unique"]}</td>'
                        f'<td>{detail}</td></tr>')
        parts.append('</table>')

    parts.append('</div>')
    return "\n".join(parts)


def _build_findings_section(findings: list) -> str:
    parts = ['<h2>Data Quality Findings</h2>']
    for f in findings:
        sev = f.get("severity", "low")
        parts.append(f'<div class="finding finding-{sev}">')
        parts.append(f'<span class="severity-badge sev-{sev}">{sev.upper()}</span> ')
        parts.append(f'<strong>{_escape(f.get("title", ""))}</strong>')
        parts.append(f'<p>{_escape(f.get("description", ""))}</p>')
        rec = f.get("recommendation", "")
        if rec:
            parts.append(f'<p><em>Recommendation: {_escape(rec)}</em></p>')
        parts.append('</div>')
    return "\n".join(parts)


def _build_stats_section(raw: dict) -> str:
    parts = []

    normality = raw.get("normality_tests", [])
    if normality:
        parts.append('<h3>Normality Tests</h3>')
        parts.append('<table><tr><th>Column</th><th>Statistic</th><th>p-value</th><th>Normal?</th><th>Skewness</th></tr>')
        for t in normality:
            normal_str = "Yes" if t["is_normal"] else "No"
            parts.append(f'<tr><td>{t["column"]}</td><td>{t["statistic"]}</td>'
                        f'<td>{t["p_value"]}</td><td>{normal_str}</td><td>{t["skewness"]}</td></tr>')
        parts.append('</table>')

    chi_sq = raw.get("chi_square_tests", [])
    if chi_sq:
        parts.append('<h3>Chi-Square Tests</h3>')
        parts.append('<table><tr><th>Columns</th><th>Statistic</th><th>p-value</th><th>Significant?</th></tr>')
        for t in chi_sq:
            sig = "Yes" if t["significant"] else "No"
            cols = " vs ".join(t["columns"])
            parts.append(f'<tr><td>{cols}</td><td>{t["statistic"]}</td>'
                        f'<td>{t["p_value"]}</td><td>{sig}</td></tr>')
        parts.append('</table>')

    return "\n".join(parts)
