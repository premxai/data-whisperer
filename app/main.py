import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import numpy as np
import polars as pl
import duckdb

from app.utils.data_loader import load_file, get_metadata, detect_file_type
from app.utils.llm_client import check_connection as check_llm, get_provider_info
from app.utils.embeddings import query_dataset, get_chroma_client
from app.agents.orchestrator import run_analysis

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
UPLOAD_DIR = os.path.join(OUTPUT_DIR, "uploads")
MAX_AGE_HOURS = 24

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "logs"), exist_ok=True)

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, "logs", "app.log")),
    ],
)

app = FastAPI(title="DataWhisperer", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# in-memory store for analysis status and results
# in production you'd use Redis or a database
analysis_store: dict[str, dict] = {}


def _sanitize(obj):
    """Recursively convert numpy/non-serializable types to native Python."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@app.get("/health")
def health_check():
    ollama_ok = False
    try:
        ollama_ok = check_llm()
    except Exception:
        pass

    chroma_ok = False
    try:
        get_chroma_client()
        chroma_ok = True
    except Exception:
        pass

    llm_info = get_provider_info()
    return {
        "status": "ok",
        "llm": {
            "connected": ollama_ok,
            "provider": llm_info["provider"],
            "model": llm_info["model"],
            "privacy": llm_info["privacy"],
        },
        "chromadb": "connected" if chroma_ok else "unavailable",
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    upload_id = str(uuid.uuid4())[:12]

    ext = Path(file.filename).suffix.lower()
    try:
        detect_file_type(file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    save_path = os.path.join(UPLOAD_DIR, f"{upload_id}{ext}")

    try:
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        logger.error(f"Failed to save upload: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")

    # quick validation
    try:
        df = load_file(save_path)
        meta = get_metadata(save_path, df)
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(status_code=400, detail=f"Invalid file: {e}")

    analysis_store[upload_id] = {
        "status": "uploaded",
        "metadata": meta,
        "file_path": save_path,
        "created_at": time.time(),
    }

    logger.info(f"File uploaded: {file.filename} -> {upload_id}")

    return {
        "upload_id": upload_id,
        "filename": file.filename,
        "metadata": meta,
    }


@app.post("/analyze/{upload_id}")
async def start_analysis(upload_id: str, background_tasks: BackgroundTasks):
    if upload_id not in analysis_store:
        raise HTTPException(status_code=404, detail="Upload not found")

    entry = analysis_store[upload_id]
    if entry["status"] not in ("uploaded", "error", "complete"):
        return {"upload_id": upload_id, "status": entry["status"],
                "message": "Analysis already in progress"}

    entry["status"] = "running"
    background_tasks.add_task(_run_analysis_task, upload_id)

    return {"upload_id": upload_id, "status": "running"}


def _run_analysis_task(upload_id: str):
    """Background task that runs the full analysis pipeline."""
    entry = analysis_store.get(upload_id)
    if not entry:
        return

    try:
        df = load_file(entry["file_path"])
        result = run_analysis(upload_id, df, entry["metadata"])

        entry["results"] = result
        entry["status"] = result.get("status", "complete")

        # generate exports
        try:
            from app.exports.html_gen import generate_html
            html_path = generate_html(upload_id, result, OUTPUT_DIR)
            entry["html_path"] = html_path
        except Exception as e:
            logger.warning(f"HTML export failed: {e}")

        try:
            from app.exports.pdf_gen import generate_pdf
            pdf_path = generate_pdf(upload_id, result, OUTPUT_DIR)
            entry["pdf_path"] = pdf_path
        except Exception as e:
            logger.warning(f"PDF export failed: {e}")

    except Exception as e:
        logger.error(f"Analysis failed for {upload_id}: {e}", exc_info=True)
        entry["status"] = "error"
        entry["error"] = str(e)


@app.get("/status/{upload_id}")
def get_status(upload_id: str):
    if upload_id not in analysis_store:
        raise HTTPException(status_code=404, detail="Upload not found")

    entry = analysis_store[upload_id]
    return {
        "upload_id": upload_id,
        "status": entry["status"],
        "error": entry.get("error"),
    }


@app.get("/results/{upload_id}")
def get_results(upload_id: str):
    if upload_id not in analysis_store:
        raise HTTPException(status_code=404, detail="Upload not found")

    entry = analysis_store[upload_id]
    if entry["status"] != "complete":
        raise HTTPException(status_code=202, detail=f"Analysis status: {entry['status']}")

    results = entry.get("results", {})

    # strip non-serializable items (plotly figures stored separately)
    safe_results = {}
    for key, val in results.items():
        if key == "visualizations":
            safe_results["visualizations"] = [
                {"type": v["type"], "title": v["title"], "reason": v.get("reason", "")}
                for v in (val or [])
            ]
        elif key == "start_time":
            continue
        else:
            safe_results[key] = val

    return _sanitize({
        "upload_id": upload_id,
        "metadata": entry.get("metadata"),
        "results": safe_results,
    })


@app.get("/download/{upload_id}")
def download_report(upload_id: str, format: str = "html"):
    if upload_id not in analysis_store:
        raise HTTPException(status_code=404, detail="Upload not found")

    entry = analysis_store[upload_id]

    if format == "pdf":
        path = entry.get("pdf_path")
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="PDF not available")
        return FileResponse(path, media_type="application/pdf",
                          filename=f"datawhisperer_report_{upload_id}.pdf")

    path = entry.get("html_path")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="HTML report not available")
    return FileResponse(path, media_type="text/html",
                       filename=f"datawhisperer_report_{upload_id}.html")


@app.post("/chat/{upload_id}")
async def chat_with_data(upload_id: str, question: dict):
    """Answer questions about the dataset using RAG + DuckDB."""
    if upload_id not in analysis_store:
        raise HTTPException(status_code=404, detail="Upload not found")

    entry = analysis_store[upload_id]
    q = question.get("question", "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="No question provided")

    # retrieve context from ChromaDB
    context_docs = query_dataset(upload_id, q)
    context = "\n".join(context_docs) if context_docs else "No context available."

    # try to answer with SQL if it looks like a data question
    sql_result = None
    file_path = entry.get("file_path")
    if file_path and os.path.exists(file_path):
        sql_result = _try_sql_query(file_path, q, entry.get("metadata", {}))

    # build prompt
    from app.utils.llm_client import generate
    prompt = (
        f"Context about the dataset:\n{context}\n\n"
    )
    if sql_result:
        prompt += f"SQL query result:\n{sql_result}\n\n"
    prompt += f"User question: {q}\n\nProvide a clear, concise answer."

    try:
        answer = generate(prompt, system="You are a helpful data analyst. Answer based on the provided context.")
    except ConnectionError:
        if sql_result:
            answer = f"SQL result: {sql_result}"
        else:
            answer = "LLM is unavailable. Please check that Ollama is running."

    return {"answer": answer, "sql_result": sql_result}


def _try_sql_query(file_path: str, question: str, metadata: dict) -> Optional[dict]:
    """Attempt to generate and execute a SQL query for the question."""
    from app.utils.llm_client import generate

    cols = metadata.get("column_names", [])
    dtypes = metadata.get("dtypes", {})

    col_desc = ", ".join(f"{c} ({dtypes.get(c, 'unknown')})" for c in cols)

    prompt = (
        f"Given a table 'data' with columns: {col_desc}\n"
        f"Write a SQL query to answer: {question}\n"
        f"Return ONLY the SQL query, nothing else. Use DuckDB SQL syntax."
    )

    try:
        sql = generate(prompt, temperature=0.1, max_tokens=256)
        sql = sql.strip().strip("`").strip()
        if sql.lower().startswith("sql"):
            sql = sql[3:].strip()

        # safety: only allow SELECT
        if not sql.upper().startswith("SELECT"):
            return None

        con = duckdb.connect()
        con.execute(f"CREATE TABLE data AS SELECT * FROM read_csv_auto('{file_path}')")
        result = con.execute(sql).fetchall()
        columns = [desc[0] for desc in con.description]
        con.close()

        # format result
        rows = [dict(zip(columns, row)) for row in result[:20]]
        return {"sql": sql, "rows": rows, "columns": columns}

    except Exception as e:
        logger.debug(f"SQL query attempt failed: {e}")
        return None


@app.on_event("startup")
async def startup_cleanup():
    """Clean up old uploads on startup."""
    _cleanup_old_files()


def _cleanup_old_files():
    now = time.time()
    cutoff = now - (MAX_AGE_HOURS * 3600)

    if not os.path.exists(UPLOAD_DIR):
        return

    for f in os.listdir(UPLOAD_DIR):
        path = os.path.join(UPLOAD_DIR, f)
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                logger.info(f"Cleaned up old file: {f}")
        except Exception:
            pass
