# src/agent/tools.py
import os
import json
from typing import List, Dict, Any, Tuple
from langchain_core.tools import tool
from langchain_core.documents import Document

# Soft cap for how much text we send back to the agent from tools
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))

def _dedup_citations(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate citations by (source, chunk, doc_title, corpus)."""
    seen: set[Tuple[str, int, str, str]] = set()
    out: List[Dict[str, Any]] = []
    for it in items:
        key = (
            str(it.get("source", "")),
            int(it.get("chunk", 0)),
            str(it.get("doc_title", "")),
            str(it.get("corpus", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out

def _pack(docs: List[Document]) -> str:
    """
    Return a JSON string payload with:
      {
        "context": "<joined compact blocks>",
        "citations": [{"doc_title","chunk","source","corpus"}, ...]
      }
    The context is trimmed by MAX_CONTEXT_CHARS to avoid oversized prompts.
    """
    if not docs:
        return json.dumps({"context": "", "citations": []}, ensure_ascii=False)

    ctx_blocks: List[str] = []
    citations: List[Dict[str, Any]] = []

    for d in docs:
        title = d.metadata.get("doc_title", "unknown")
        chunk = d.metadata.get("chunk", 0)
        corpus = d.metadata.get("corpus", "")
        src = d.metadata.get("source", "")
        text = d.page_content or ""
        ctx_blocks.append(f"[{title} :: #{chunk}]\n{text}")
        citations.append({
            "doc_title": title,
            "chunk": chunk,
            "source": src,
            "corpus": corpus,
        })

    # Join and trim the context (keep head, it’s usually most relevant)
    context = "\n\n".join(ctx_blocks)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    citations = _dedup_citations(citations)
    payload = {"context": context, "citations": citations}
    return json.dumps(payload, ensure_ascii=False)

def _coerce_k(top_k: Any, default_k: int) -> int:
    try:
        if top_k is None:
            return int(default_k)
        return max(1, int(top_k))
    except Exception:
        return int(default_k)

def build_tools(retriever, default_k: int):
    @tool("hr_search", return_direct=False)
    def hr_search(query: str, top_k: int | str | None = None) -> str:
        """
        ابحث فقط في مصدر سياسات الموارد البشرية (HR).
        Args:
          query: نص السؤال.
          top_k: عدد النتائج المراد استرجاعها (اختياري).
        Returns:
          JSON string:
          {
            "context": "...",
            "citations": [
              {"doc_title": str, "chunk": int, "source": str, "corpus": "hr"},
              ...
            ]
          }
        """
        k = _coerce_k(top_k, default_k)
        docs = retriever.get_relevant_documents(query, k=k, filter={"corpus": "hr"})
        return _pack(docs)

    @tool("jisr_search", return_direct=False)
    def jisr_search(query: str, top_k: int | str | None = None) -> str:
        """
        ابحث فقط في مصدر أدلة منصة جسر (JISR).
        Args:
          query: نص السؤال.
          top_k: عدد النتائج المراد استرجاعها (اختياري).
        Returns:
          JSON string (نفس صيغة hr_search) ولكن corpus = "jisr".
        """
        k = _coerce_k(top_k, default_k)
        docs = retriever.get_relevant_documents(query, k=k, filter={"corpus": "jisr"})
        return _pack(docs)

    return [hr_search, jisr_search]
