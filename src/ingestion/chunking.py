from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(text: str, max_tokens=800, overlap=120):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens * 4,
        chunk_overlap=overlap * 4,
        separators=["\n\n", "\n", ".", "،", " "],
    )
    return splitter.split_text(text or "")
