import logging

import numpy as np
import polars as pl
from scipy import stats as scipy_stats

from app.utils.llm_client import generate_json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a statistician agent. Given statistical test results from a dataset,
interpret them in plain English. Explain what the numbers mean for someone who isn't a stats expert.

Respond in JSON:
{
    "interpretations": [
        {
            "test_name": "name of test",
            "columns": ["col1"],
            "result_summary": "plain English interpretation",
            "significance": "significant|not_significant|borderline",
            "practical_meaning": "what this means for the data"
        }
    ],
    "key_takeaways": ["takeaway 1", "takeaway 2"]
}"""


def normality_tests(df: pl.DataFrame, numeric_cols: list[str]) -> list[dict]:
    """Run Shapiro-Wilk normality test on numeric columns."""
    results = []
    for col in numeric_cols:
        values = df[col].drop_nulls().to_numpy().astype(np.float64)
        if len(values) < 8 or len(values) > 5000:
            # shapiro doesn't work well outside this range
            if len(values) > 5000:
                values = np.random.choice(values, 5000, replace=False)
            elif len(values) < 8:
                continue

        try:
            stat, p_value = scipy_stats.shapiro(values)
            skew = float(scipy_stats.skew(values))
            kurt = float(scipy_stats.kurtosis(values))

            results.append({
                "column": col,
                "test": "shapiro_wilk",
                "statistic": round(float(stat), 6),
                "p_value": round(float(p_value), 6),
                "is_normal": p_value > 0.05,
                "skewness": round(skew, 4),
                "kurtosis": round(kurt, 4),
            })
        except Exception as e:
            logger.debug(f"Normality test failed for {col}: {e}")

    return results


def chi_square_tests(df: pl.DataFrame, categorical_cols: list[str]) -> list[dict]:
    """Run chi-square independence tests between pairs of categorical columns."""
    from itertools import combinations

    results = []
    cols = categorical_cols[:10]  # limit pairs

    for col_a, col_b in combinations(cols, 2):
        try:
            a = df[col_a].drop_nulls().to_list()
            b = df[col_b].drop_nulls().to_list()

            # align on non-null rows
            mask = df[col_a].is_not_null() & df[col_b].is_not_null()
            a = df.filter(mask)[col_a].to_list()
            b = df.filter(mask)[col_b].to_list()

            if len(a) < 10:
                continue

            # build contingency table
            a_vals = sorted(set(a))
            b_vals = sorted(set(b))
            if len(a_vals) < 2 or len(b_vals) < 2:
                continue
            if len(a_vals) > 20 or len(b_vals) > 20:
                continue

            a_map = {v: i for i, v in enumerate(a_vals)}
            b_map = {v: i for i, v in enumerate(b_vals)}
            table = np.zeros((len(a_vals), len(b_vals)))
            for av, bv in zip(a, b):
                table[a_map[av], b_map[bv]] += 1

            chi2, p_value, dof, expected = scipy_stats.chi2_contingency(table)

            results.append({
                "columns": [col_a, col_b],
                "test": "chi_square",
                "statistic": round(float(chi2), 4),
                "p_value": round(float(p_value), 6),
                "dof": int(dof),
                "significant": p_value < 0.05,
            })
        except Exception as e:
            logger.debug(f"Chi-square test failed for {col_a} x {col_b}: {e}")

    return results


def distribution_comparison(df: pl.DataFrame, numeric_cols: list[str]) -> list[dict]:
    """Check if numeric distributions differ significantly from uniform/normal."""
    results = []
    for col in numeric_cols[:15]:
        values = df[col].drop_nulls().to_numpy().astype(np.float64)
        if len(values) < 20:
            continue

        # Anderson-Darling test for normality
        try:
            ad_result = scipy_stats.anderson(values, dist="norm")
            results.append({
                "column": col,
                "test": "anderson_darling",
                "statistic": round(float(ad_result.statistic), 4),
                "critical_values": {
                    f"{sl}%": round(float(cv), 4)
                    for sl, cv in zip(ad_result.significance_level, ad_result.critical_values)
                },
                "likely_normal": ad_result.statistic < ad_result.critical_values[2],
            })
        except Exception as e:
            logger.debug(f"Anderson-Darling failed for {col}: {e}")

    return results


def run(df: pl.DataFrame, profile: dict, correlations: dict) -> dict:
    """Run statistical analysis and get LLM interpretations."""
    logger.info("Statistician agent starting analysis")

    column_types = profile.get("columns", {})
    numeric_cols = [c for c, info in column_types.items() if info["detected_type"] == "numeric"]
    categorical_cols = [c for c, info in column_types.items() if info["detected_type"] == "categorical"]

    # run tests
    normality = normality_tests(df, numeric_cols)
    chi_sq = chi_square_tests(df, categorical_cols)
    distributions = distribution_comparison(df, numeric_cols)

    raw_results = {
        "normality_tests": normality,
        "chi_square_tests": chi_sq,
        "distribution_tests": distributions,
        "high_correlations": correlations.get("numeric", {}).get("high_correlations", []),
    }

    # get LLM interpretation
    context = _format_results(raw_results)
    prompt = f"Interpret these statistical results:\n\n{context}"

    try:
        interpretation = generate_json(prompt, system=SYSTEM_PROMPT, max_tokens=2048)
        if interpretation.get("parse_error"):
            interpretation = _fallback_interpretation(raw_results)
    except ConnectionError:
        logger.warning("LLM unavailable for statistician, using fallback")
        interpretation = _fallback_interpretation(raw_results)

    logger.info("Statistician agent complete")
    return {
        "raw_results": raw_results,
        "interpretation": interpretation,
    }


def _format_results(results: dict) -> str:
    lines = []

    normality = results.get("normality_tests", [])
    if normality:
        lines.append("Normality Tests (Shapiro-Wilk):")
        for t in normality:
            status = "normal" if t["is_normal"] else "not normal"
            lines.append(f"  {t['column']}: p={t['p_value']} ({status}), skew={t['skewness']}")

    chi_sq = results.get("chi_square_tests", [])
    if chi_sq:
        lines.append("\nChi-Square Independence Tests:")
        for t in chi_sq:
            sig = "significant" if t["significant"] else "not significant"
            lines.append(f"  {t['columns'][0]} vs {t['columns'][1]}: p={t['p_value']} ({sig})")

    high_corr = results.get("high_correlations", [])
    if high_corr:
        lines.append("\nHigh Correlations:")
        for c in high_corr:
            lines.append(f"  {c['col1']} <-> {c['col2']}: r={c['correlation']}")

    return "\n".join(lines)


def _fallback_interpretation(results: dict) -> dict:
    interpretations = []

    for t in results.get("normality_tests", []):
        interpretations.append({
            "test_name": "Shapiro-Wilk normality",
            "columns": [t["column"]],
            "result_summary": (
                f"{'Appears normally distributed' if t['is_normal'] else 'Not normally distributed'} "
                f"(p={t['p_value']}). Skewness: {t['skewness']}."
            ),
            "significance": "not_significant" if t["is_normal"] else "significant",
            "practical_meaning": (
                "Parametric tests are appropriate." if t["is_normal"]
                else "Consider non-parametric methods or transformations."
            ),
        })

    for t in results.get("chi_square_tests", []):
        interpretations.append({
            "test_name": "Chi-square independence",
            "columns": t["columns"],
            "result_summary": (
                f"{'Statistically significant' if t['significant'] else 'No significant'} "
                f"relationship found (p={t['p_value']})."
            ),
            "significance": "significant" if t["significant"] else "not_significant",
            "practical_meaning": (
                "These variables appear to be related." if t["significant"]
                else "No evidence of association between these variables."
            ),
        })

    takeaways = []
    n_normal = sum(1 for t in results.get("normality_tests", []) if t["is_normal"])
    n_total = len(results.get("normality_tests", []))
    if n_total > 0:
        takeaways.append(f"{n_normal}/{n_total} numeric columns follow a normal distribution.")

    n_sig = sum(1 for t in results.get("chi_square_tests", []) if t["significant"])
    if results.get("chi_square_tests"):
        takeaways.append(f"{n_sig} significant categorical relationships found.")

    return {
        "interpretations": interpretations,
        "key_takeaways": takeaways,
    }
