import logging
import os
from typing import Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore")
COLLECTION_NAME = "dataset_knowledge"

_chroma_client = None
_collection = None


def get_chroma_client() -> chromadb.Client:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        logger.info(f"ChromaDB initialized at {CHROMA_PERSIST_DIR}")
    return _chroma_client


def get_collection(name: str = COLLECTION_NAME) -> chromadb.Collection:
    global _collection
    if _collection is None or _collection.name != name:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Using collection '{name}' ({_collection.count()} docs)")
    return _collection


def embed_dataset_schema(upload_id: str, metadata: dict, profile: dict, sample_rows: list[dict]):
    """Store dataset schema and profile info in ChromaDB for RAG."""
    collection = get_collection()

    documents = []
    ids = []
    metadatas = []

    # schema overview
    col_descriptions = []
    for col_name, col_info in profile.get("columns", {}).items():
        desc = f"Column '{col_name}': type={col_info['detected_type']}, dtype={col_info['dtype']}"
        if col_info.get("stats"):
            stats = col_info["stats"]
            if "mean" in stats:
                desc += f", mean={stats['mean']}, min={stats.get('min')}, max={stats.get('max')}"
            elif "n_unique" in stats:
                desc += f", {stats['n_unique']} unique values"
        col_descriptions.append(desc)

    schema_doc = (
        f"Dataset: {metadata.get('filename', 'unknown')}\n"
        f"Shape: {metadata.get('rows')} rows x {metadata.get('columns')} columns\n"
        f"Columns:\n" + "\n".join(col_descriptions)
    )

    documents.append(schema_doc)
    ids.append(f"{upload_id}_schema")
    metadatas.append({"upload_id": upload_id, "type": "schema"})

    # profile summary
    profile_doc = (
        f"Data quality: {profile.get('completeness_pct', 0)}% complete, "
        f"{profile.get('total_nulls', 0)} total nulls, "
        f"{profile.get('duplicate_rows', 0)} duplicate rows. "
        f"Type breakdown: {profile.get('type_counts', {})}"
    )
    documents.append(profile_doc)
    ids.append(f"{upload_id}_profile")
    metadatas.append({"upload_id": upload_id, "type": "profile"})

    # sample rows
    if sample_rows:
        sample_text = "Sample data rows:\n"
        for i, row in enumerate(sample_rows[:5]):
            sample_text += f"Row {i}: {row}\n"
        documents.append(sample_text)
        ids.append(f"{upload_id}_samples")
        metadatas.append({"upload_id": upload_id, "type": "samples"})

    collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
    logger.info(f"Embedded {len(documents)} documents for upload {upload_id}")


def query_dataset(upload_id: str, question: str, n_results: int = 5) -> list[str]:
    """Retrieve relevant context for a question about the dataset."""
    collection = get_collection()

    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        where={"upload_id": upload_id},
    )

    docs = results.get("documents", [[]])[0]
    return docs


def clear_dataset(upload_id: str):
    """Remove all stored data for a given upload."""
    collection = get_collection()
    try:
        collection.delete(where={"upload_id": upload_id})
        logger.info(f"Cleared embeddings for upload {upload_id}")
    except Exception as e:
        logger.warning(f"Could not clear embeddings for {upload_id}: {e}")
