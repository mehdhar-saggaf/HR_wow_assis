# src/ingestion/ingest_pipeline.py
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple

from src.ingestion.loaders import load_documents
from src.ingestion.cleaning import clean_document
from src.ingestion.chunking import chunk_text
from src.rag.embeddings import get_embeddings
from src.rag.store import get_vector_store
from src.config import Settings

# Raw sources
RAW_POLICIES = Path("data/raw/hr_policies")
RAW_JISR = Path("data/raw/jisr_guides")


def _infer_corpus(src_path: str) -> str:
    """Infer corpus label ('hr' | 'jisr' | 'unknown') from the absolute/relative path."""
    p = (src_path or "").lower()
    if "data/raw/hr_policies" in p or "hr_policies" in p:
        return "hr"
    if "data/raw/jisr_guides" in p or "jisr_guides" in p:
        return "jisr"
    return "unknown"


def run_ingestion(settings: Settings, source: str = "all"):
    """
    Ingest documents from the requested sources, clean, chunk, and store in Chroma.

    Args:
        settings: global Settings object
        source: "all" | "policies" | "jisr"

    Returns:
        dict with ingestion stats, including per-corpus counts.
    """
    # 1) Collect folders to scan
    folders: List[Path] = []
    if source in ("all", "policies"):
        folders.append(RAW_POLICIES)
    if source in ("all", "jisr"):
        folders.append(RAW_JISR)

    # 2) Load files (pdf/docx/txt/md)
    all_docs = []
    for fld in folders:
        all_docs.extend(load_documents(fld))

    # 3) Normalize/clean text (Arabic normalization, etc.)
    cleaned = [clean_document(d) for d in all_docs]

    # 4) Chunk and tag metadata
    records: List[Tuple[str, dict]] = []
    corpus_counts: Dict[str, int] = {"hr": 0, "jisr": 0, "unknown": 0}

    for d in tqdm(cleaned, desc="Chunking"):
        text = d.get("text", "") or ""
        meta = d.get("meta", {}) or {}

        chunks = chunk_text(
            text,
            settings.MAX_CHUNK_TOKENS,
            settings.CHUNK_OVERLAP,
        )

        # Prefer loader-provided corpus; fallback to path inference
        src_path = meta.get("source", "")
        corpus = meta.get("corpus") or _infer_corpus(src_path)

        for i, ch in enumerate(chunks):
            ch = (ch or "").strip()
            if not ch:
                continue  # skip empty after cleaning/splitting

            # Preserve original metadata and add chunk index + corpus tag
            ch_meta = dict(meta) | {"chunk": i, "corpus": corpus}
            records.append((ch, ch_meta))
            corpus_counts[corpus] = corpus_counts.get(corpus, 0) + 1

    if not records:
        return {
            "ingested": 0,
            "files": len(all_docs),
            "by_corpus": corpus_counts,
            "source": source,
        }

    # 5) Get embeddings + vector store
    embeddings = get_embeddings(settings)
    vs = get_vector_store(settings, embeddings)

    # 6) Add to Chroma
    texts = [r[0] for r in records]
    metadatas = [r[1] for r in records]
    vs.add_texts(texts=texts, metadatas=metadatas)

    # 7) Persist to disk (best effort)
    try:
        vs.persist()
    except Exception:
        # Chroma may persist automatically; ignore soft failures here
        pass

    # 8) Return stats
    return {
        "ingested": len(texts),
        "files": len(all_docs),
        "by_corpus": corpus_counts,
        "source": source,
    }
