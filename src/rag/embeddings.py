# src/rag/embeddings.py
import logging
from typing import Optional, Dict, Any, List

from src.config import Settings

logger = logging.getLogger(__name__)

# Prefer the new package; fall back to old if not installed
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:  # pragma: no cover
    from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore


def _is_e5(model_name: str) -> bool:
    mn = (model_name or "").lower()
    return "e5" in mn  # e.g., "intfloat/multilingual-e5-base"


def _build_kwargs(model_name: str) -> Dict[str, Any]:
    """
    Common kwargs for HF embeddings.
    - normalize_embeddings=True helps cosine similarity.
    - device="cpu" by default (switch to "cuda" if you have a GPU).
    """
    return {
        "model_name": model_name,
        "encode_kwargs": {"normalize_embeddings": True},
        "model_kwargs": {"device": "cpu"},
    }


class _E5Embeddings(HuggingFaceEmbeddings):
    """
    Wrap HuggingFaceEmbeddings to add E5 instructions transparently.
    E5 expects:
      - embed_query:  'query: <text>'
      - embed_docs:   'passage: <text>'
    """
    def embed_documents(self, texts: List[str], **kwargs) -> List[float]:
        texts = [f"passage: {t}" for t in texts]
        return super().embed_documents(texts, **kwargs)

    def embed_query(self, text: str, **kwargs) -> List[float]:
        return super().embed_query(f"query: {text}", **kwargs)


_cached_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings(settings: Settings):
    """
    Returns a singleton embeddings object.
    - If HF_MODEL contains 'e5', we use the _E5Embeddings wrapper.
    - Otherwise, we use vanilla HuggingFaceEmbeddings.
    """
    global _cached_embeddings
    if _cached_embeddings is not None:
        return _cached_embeddings

    model_name = settings.HF_MODEL
    logger.info(f"Initializing HuggingFace embeddings: {model_name}")

    kwargs = _build_kwargs(model_name)
    if _is_e5(model_name):
        _cached_embeddings = _E5Embeddings(**kwargs)
        logger.info("Initialized E5 wrapper embeddings")
    else:
        _cached_embeddings = HuggingFaceEmbeddings(**kwargs)
        logger.info("Initialized standard HF embeddings")

    logger.info("Embeddings initialized successfully")
    return _cached_embeddings
