import os
import logging
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from dotenv import load_dotenv

from src.config import Settings
from src.utils.logging import setup_logging
from src.rag.store import initialize_vector_store, clear_vector_store
from src.rag.retrieval import build_retriever

# NEW: agent + chat history message types
from langchain_core.messages import HumanMessage, AIMessage
from src.agent.hr_agent import build_hr_agent

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

# If your index.html lives under ./templates (as in this project), keep as-is.
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

settings = Settings()

# ---- Vector store & retriever ------------------------------------------------
vector_store = initialize_vector_store(settings)
logger.info("Vector store initialized successfully")

retriever = build_retriever(vector_store, settings)

# ---- hr_agent (tool-calling) -------------------------------------------------
agent_executor = None
try:
    agent_executor = build_hr_agent(retriever, settings)
    logger.info("hr_agent initialized")
except Exception as e:
    logger.warning(f"hr_agent init failed (likely missing/invalid Groq API key): {e}")
    logger.info("Running in fallback mode - simple document retrieval will be used")

# ---- In-memory session history (per session_id) ------------------------------
_SESSIONS: dict[str, list] = {}

def _get_history(session_id: str) -> list:
    """Return (and create if missing) message history list for a session."""
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = []
    return _SESSIONS[session_id]

# ---- Small-talk shortcut -----------------------------------------------------
SMALLTALK_KEYWORDS = {
    "hi", "hello", "hey", "hii", "good morning", "good evening", "good night",
    "مرحبا", "السلام عليكم", "هلا", "هاي", "صباح الخير", "مساء الخير",
    "كيف حالك", "شلونك", "كيفك", "اخبارك"
}

def _is_smalltalk(msg: str) -> bool:
    m = msg.strip().lower()
    for kw in SMALLTALK_KEYWORDS:
        if kw in m:
            return True
    return False

def _smalltalk_reply(msg: str) -> str:
    return "مرحبًا! كيف أقدر أساعدك اليوم؟ 😊"

# ---- Routes ------------------------------------------------------------------
@app.get("/")
def home():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/reset")
def reset_store():
    """Clear Chroma collection and filesystem directory safely."""
    ok = clear_vector_store(settings)
    return jsonify({"ok": ok})

@app.post("/ingest")
def ingest():
    from src.ingestion.ingest_pipeline import run_ingestion
    payload = request.get_json(silent=True) or {}
    source = payload.get("source", "all")  # "all" | "policies" | "jisr"
    stats = run_ingestion(settings, source)
    return jsonify({"ok": True, "stats": stats})

@app.post("/chat")
def chat():
    payload = request.get_json(force=True)
    msg = (payload.get("message") or "").strip()
    top_k = int(payload.get("top_k", settings.DEFAULT_TOP_K))
    session_id = payload.get("session_id", "default")

    if not msg:
        return jsonify({"answer": "يرجى كتابة سؤالك.", "citations": []})

    # 👉 Small-talk fast path
    if _is_smalltalk(msg):
        return jsonify({"answer": _smalltalk_reply(msg), "citations": []})

    try:
        logger.info(f"[{session_id}] user: {msg[:120]}")

        # Prefer agent path
        if agent_executor is not None:
            history = _get_history(session_id)

            result = agent_executor.invoke({
                "input": msg,
                "chat_history": history,  # list[BaseMessage]
            })
            output = result.get("output", "")

            # Extract structured citations from the model's tagged JSON block
            import re, json as _json
            citations = []
            m = re.search(r"<citations>(.*?)</citations>", output, flags=re.S)
            if m:
                try:
                    citations = _json.loads(m.group(1)).get("items", [])
                    output = output.replace(m.group(0), "").strip()
                except Exception:
                    pass

            # Save to history
            history.append(HumanMessage(content=msg))
            history.append(AIMessage(content=output))

            return jsonify({"answer": output, "citations": citations})

        # ---- Fallback: simple retrieval only --------------------------------
        logger.info("Agent unavailable; using simple retriever fallback")
        docs = retriever.get_relevant_documents(msg, k=top_k)
        if docs:
            chunks = []
            for d in docs[:3]:
                title = d.metadata.get("doc_title", "غير معروف")
                chunk_idx = d.metadata.get("chunk", 0)
                text = (d.page_content or "")[:400]
                chunks.append(f"- [{title} :: #{chunk_idx}]\n{text}")
            answer = "ملخص من المصادر (الوضع الاحتياطي):\n\n" + "\n\n".join(chunks)
            citations = [{
                "doc_title": d.metadata.get("doc_title", "unknown"),
                "chunk": d.metadata.get("chunk", 0),
                "source": d.metadata.get("source", ""),
                "corpus": d.metadata.get("corpus", ""),
            } for d in docs]
        else:
            answer = "لا توجد مصادر كافية للإجابة حالياً."
            citations = []

        return jsonify({"answer": answer, "citations": citations})

    except Exception as e:
        logger.exception("Error handling /chat")
        return jsonify({"error": str(e)}), 500

# ---- Main --------------------------------------------------------------------
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting HR RAG Chatbot on {host}:{port}")
    app.run(host=host, port=port, debug=True)
