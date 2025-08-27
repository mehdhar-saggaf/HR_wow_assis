from src.config import Settings

def build_retriever(vectorstore, settings: Settings):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.DEFAULT_TOP_K,
            "fetch_k": max(settings.DEFAULT_TOP_K * 3, 8),
        },
    )
    return retriever
