import logging
import os
import shutil
from typing import Optional

from langchain_chroma import Chroma
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)

_vector_store_instance: Optional[Chroma] = None

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def get_vector_store(settings, embeddings) -> Chroma:
    global _vector_store_instance

    if _vector_store_instance is None:
        logger.info(f"Initializing vector store at: {settings.CHROMA_DIR}")
        try:
            _ensure_dir(settings.CHROMA_DIR)

            client_settings = ChromaSettings(
                persist_directory=settings.CHROMA_DIR,
                anonymized_telemetry=False,
                allow_reset=True,
            )

            _vector_store_instance = Chroma(
                embedding_function=embeddings,
                persist_directory=settings.CHROMA_DIR,
                client_settings=client_settings,
                collection_name="hr_documents",
            )

            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise

    return _vector_store_instance

def initialize_vector_store(settings):
    from src.rag.embeddings import get_embeddings
    embeddings = get_embeddings(settings)
    return get_vector_store(settings, embeddings)

def get_vector_store_stats(vector_store: Chroma) -> dict:
    try:
        collection = vector_store._collection
        count = collection.count()
        return {
            "total_documents": int(count),
            "collection_name": getattr(collection, "name", "unknown"),
        }
    except Exception as e:
        logger.error(f"Error getting vector store stats: {e}")
        return {"total_documents": 0, "collection_name": "unknown"}

def clear_vector_store(settings) -> bool:
    global _vector_store_instance
    try:
        vs = initialize_vector_store(settings)
        coll_name = vs._collection.name  # type: ignore[attr-defined]
        vs._client.delete_collection(name=coll_name)  # type: ignore[attr-defined]
        _vector_store_instance = None

        if os.path.isdir(settings.CHROMA_DIR):
            shutil.rmtree(settings.CHROMA_DIR, ignore_errors=True)
            _ensure_dir(settings.CHROMA_DIR)

        logger.info("Vector store cleared successfully")
        return True
    except Exception as e:
        logger.error(f"Error clearing vector store: {e}")
        return False

def test_vector_store(settings) -> bool:
    try:
        vs = initialize_vector_store(settings)
        vs.add_texts(
            texts=["هذا نص تجريبي للاختبار"],
            metadatas=[{"source": "test", "doc_title": "test_doc"}]
        )
        results = vs.similarity_search("تجريبي", k=1)
        ok = bool(results)
        if ok:
            logger.info("Vector store test successful")
        else:
            logger.error("Vector store test failed: no results")
        return ok
    except Exception as e:
        logger.error(f"Vector store test failed: {e}")
        return False
