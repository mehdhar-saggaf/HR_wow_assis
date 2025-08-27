import os
from pydantic import BaseModel

class Settings(BaseModel):
    ANSWER_LANG: str = os.getenv("ANSWER_LANG", "ar")
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", "./vectorstore/chroma")
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    MAX_CHUNK_TOKENS: int = int(os.getenv("MAX_CHUNK_TOKENS", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    EMBEDDINGS_PROVIDER: str = os.getenv("EMBEDDINGS_PROVIDER", "hf")
    HF_MODEL: str = os.getenv("HF_MODEL", "sentence-transformers/all-MiniLM-L12-v2")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
