# HR RAG Chatbot (MVP)

Arabic-first HR assistant using RAG with **Chroma**, **LangChain 0.3**, **Groq**, **LangSmith**, and **Conversation Buffer Memory**.

## Quick Start

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Fill GROQ_API_KEY, LANGCHAIN_API_KEY, etc.

# Add docs
mkdir -p data/raw/hr_policies data/raw/jisr_guides
# Put your files there

python app.py
```

In another terminal, ingest:
```bash
curl -X POST http://localhost:8000/ingest   -H "Content-Type: application/json"   -d "{\"source\":\"all\"}"
```

Open: http://localhost:8000/

## Notes
- Uses `langchain-chroma` (no deprecation warnings).
- Disable Chroma telemetry via code and `.env`.
- LangSmith tracing is enabled via environment variables.
