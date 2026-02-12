import os
import time

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="DataWhisperer", layout="wide", initial_sidebar_state="collapsed")

st.title("DataWhisperer")
st.caption("Automated exploratory data analysis powered by LLMs")


def api_get(path: str, **kwargs):
    try:
        r = requests.get(f"{API_URL}{path}", **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Is the API running?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, **kwargs):
    try:
        r = requests.post(f"{API_URL}{path}", **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend. Is the API running?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# -- sidebar health check --
with st.sidebar:
    st.header("System Status")
    if st.button("Check Health"):
        health = api_get("/health")
        if health:
            st.write("**API:** OK")

            llm = health.get("llm", {})
            provider = llm.get("provider", "unknown")
            connected = llm.get("connected", False)
            model = llm.get("model", "unknown")
            privacy = llm.get("privacy", "unknown")

            status_icon = "Connected" if connected else "Not connected (using fallbacks)"
            st.write(f"**LLM:** {provider} / {model}")
            st.write(f"**LLM Status:** {status_icon}")

            if provider == "ollama":
                st.success(f"Privacy: {privacy}")
            else:
                st.warning(f"Privacy: {privacy}")

            st.write(f"**ChromaDB:** {health.get('chromadb', 'unknown')}")

# -- session state --
if "upload_id" not in st.session_state:
    st.session_state.upload_id = None
if "results" not in st.session_state:
    st.session_state.results = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -- file upload --
st.header("Upload Dataset")

uploaded_file = st.file_uploader(
    "Choose a file (CSV, Excel, JSON, or Parquet)",
    type=["csv", "xlsx", "xls", "json", "parquet"],
)

col1, col2 = st.columns([1, 3])

with col1:
    analyze_btn = st.button("Upload & Analyze", disabled=uploaded_file is None)

if analyze_btn and uploaded_file:
    # upload
    with st.spinner("Uploading file..."):
        resp = api_post("/upload", files={"file": (uploaded_file.name, uploaded_file.getvalue())})

    if resp:
        upload_id = resp["upload_id"]
        st.session_state.upload_id = upload_id
        st.session_state.results = None
        st.session_state.chat_history = []

        meta = resp.get("metadata", {})
        st.success(f"Uploaded: {meta.get('rows', '?')} rows x {meta.get('columns', '?')} columns")

        # start analysis
        api_post(f"/analyze/{upload_id}")

        # poll for results
        progress_bar = st.progress(0, text="Analyzing...")
        status_text = st.empty()
        status_map = {
            "running": (10, "Starting analysis..."),
            "profiling_complete": (30, "Profiling complete, running detective..."),
            "detective_complete": (50, "Detective done, running statistician..."),
            "statistician_complete": (65, "Stats done, creating visualizations..."),
            "visualizer_complete": (80, "Charts done, writing report..."),
            "complete": (100, "Analysis complete!"),
        }

        max_wait = 600  # 10 minutes
        elapsed = 0
        while elapsed < max_wait:
            status_resp = api_get(f"/status/{upload_id}")
            if not status_resp:
                break

            status = status_resp.get("status", "unknown")
            pct, msg = status_map.get(status, (10, f"Status: {status}"))
            progress_bar.progress(pct, text=msg)
            status_text.text(msg)

            if status == "complete":
                break
            if status == "error":
                st.error(f"Analysis failed: {status_resp.get('error', 'Unknown error')}")
                break

            time.sleep(2)
            elapsed += 2

        if elapsed >= max_wait:
            st.warning("Analysis is taking longer than expected. Check back later.")

        # fetch results
        results_resp = api_get(f"/results/{upload_id}")
        if results_resp:
            st.session_state.results = results_resp

# -- display results --
if st.session_state.results:
    results = st.session_state.results
    upload_id = st.session_state.upload_id
    data = results.get("results", {})
    metadata = results.get("metadata", {})

    st.divider()

    # quality score
    report = data.get("report", {})
    detective_results = data.get("detective_results", {})
    quality_score = report.get("quality_score", detective_results.get("overall_quality_score", 0))

    score_col, info_col = st.columns([1, 2])
    with score_col:
        color = "green" if quality_score >= 70 else ("orange" if quality_score >= 40 else "red")
        st.metric("Data Quality Score", f"{quality_score}/100")
    with info_col:
        st.write(f"**Dataset:** {metadata.get('filename', 'Unknown')}")
        st.write(f"**Shape:** {metadata.get('rows', '?')} rows x {metadata.get('columns', '?')} columns")
        st.write(f"**Size:** {metadata.get('file_size_mb', '?')} MB")

    # tabs
    tab_profile, tab_findings, tab_viz, tab_report, tab_stats = st.tabs([
        "Data Profile", "Quality Findings", "Visualizations", "Report", "Statistics"
    ])

    with tab_profile:
        profile = data.get("profile", {})
        if profile:
            st.subheader("Overview")
            overview_cols = st.columns(4)
            overview_cols[0].metric("Completeness", f"{profile.get('completeness_pct', '?')}%")
            overview_cols[1].metric("Duplicate Rows", profile.get("duplicate_rows", 0))
            overview_cols[2].metric("Total Nulls", profile.get("total_nulls", 0))
            types = profile.get("type_counts", {})
            overview_cols[3].metric("Column Types", ", ".join(f"{v} {k}" for k, v in types.items()))

            st.subheader("Column Details")
            columns = profile.get("columns", {})
            if columns:
                table_data = []
                for name, info in columns.items():
                    stats = info.get("stats", {})
                    detail = ""
                    if "mean" in stats:
                        detail = f"mean={stats['mean']}, std={stats.get('std', '')}"
                    elif "top_values" in stats:
                        top = stats["top_values"][:3]
                        detail = ", ".join(f"{v['value']}({v['count']})" for v in top)
                    table_data.append({
                        "Column": name,
                        "Type": info["detected_type"],
                        "Null %": info["null_pct"],
                        "Unique": info["n_unique"],
                        "Details": detail,
                    })
                st.dataframe(table_data, use_container_width=True)

    with tab_findings:
        findings = detective_results.get("findings", [])
        if findings:
            for f in findings:
                sev = f.get("severity", "low")
                icon = {"high": "!!!", "medium": "!!", "low": "!"}.get(sev, "")
                with st.expander(f"[{sev.upper()}] {f.get('title', '')}", expanded=sev == "high"):
                    st.write(f.get("description", ""))
                    if f.get("affected_columns"):
                        st.write(f"**Affected columns:** {', '.join(f['affected_columns'])}")
                    if f.get("recommendation"):
                        st.info(f"**Recommendation:** {f['recommendation']}")
        else:
            st.success("No significant data quality issues found.")

    with tab_viz:
        viz_list = data.get("visualizations", [])
        if viz_list:
            st.write(f"{len(viz_list)} visualizations generated.")
            st.caption("Note: Interactive charts are available in the HTML export.")
            for v in viz_list:
                st.write(f"- **{v.get('title', '')}** ({v.get('type', '')}): {v.get('reason', '')}")
        else:
            st.info("No visualizations available. They may still be generating.")

    with tab_report:
        report_text = report.get("report_text", "")
        if report_text:
            st.write(report_text)
        else:
            st.info("Report not yet available.")

        sections = report.get("sections", {})
        if sections:
            for section_name, content in sections.items():
                with st.expander(section_name.replace("_", " ").title()):
                    st.write(content)

    with tab_stats:
        stats_data = data.get("stats_results", {})
        interp = stats_data.get("interpretation", {})
        takeaways = interp.get("key_takeaways", [])
        if takeaways:
            st.subheader("Key Takeaways")
            for t in takeaways:
                st.write(f"- {t}")

        interpretations = interp.get("interpretations", [])
        if interpretations:
            st.subheader("Statistical Interpretations")
            for si in interpretations:
                with st.expander(f"{si.get('test_name', 'Test')} - {', '.join(si.get('columns', []))}"):
                    st.write(si.get("result_summary", ""))
                    st.write(f"**Significance:** {si.get('significance', 'N/A')}")
                    st.write(f"**Practical meaning:** {si.get('practical_meaning', '')}")

    # download buttons
    st.divider()
    dl_col1, dl_col2, dl_col3 = st.columns(3)

    with dl_col1:
        if st.button("Download HTML Report"):
            try:
                r = requests.get(f"{API_URL}/download/{upload_id}?format=html")
                if r.status_code == 200:
                    st.download_button("Save HTML", r.content,
                                      file_name=f"report_{upload_id}.html",
                                      mime="text/html")
                else:
                    st.warning("HTML report not ready yet.")
            except Exception:
                st.error("Could not fetch HTML report.")

    with dl_col2:
        if st.button("Download PDF Report"):
            try:
                r = requests.get(f"{API_URL}/download/{upload_id}?format=pdf")
                if r.status_code == 200:
                    st.download_button("Save PDF", r.content,
                                      file_name=f"report_{upload_id}.pdf",
                                      mime="application/pdf")
                else:
                    st.warning("PDF report not ready yet.")
            except Exception:
                st.error("Could not fetch PDF report.")

# -- chat interface --
if st.session_state.upload_id and st.session_state.results:
    st.divider()
    st.header("Ask Questions About Your Data")
    st.caption("Ask in natural language. The system will use RAG and SQL to find answers.")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sql_result"):
                sql_info = msg["sql_result"]
                st.code(sql_info.get("sql", ""), language="sql")
                if sql_info.get("rows"):
                    st.dataframe(sql_info["rows"])

    question = st.chat_input("Ask something about your dataset...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = api_post(
                    f"/chat/{st.session_state.upload_id}",
                    json={"question": question},
                )

            if resp:
                answer = resp.get("answer", "No answer available.")
                sql_result = resp.get("sql_result")

                st.write(answer)
                if sql_result:
                    st.code(sql_result.get("sql", ""), language="sql")
                    if sql_result.get("rows"):
                        st.dataframe(sql_result["rows"])

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sql_result": sql_result,
                })
            else:
                st.error("Failed to get a response.")
