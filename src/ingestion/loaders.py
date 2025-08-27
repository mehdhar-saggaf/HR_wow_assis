# src/ingestion/loaders.py
from pathlib import Path
import re

# Try PyMuPDF (fitz) first for better Arabic extraction; fallback to pdfminer
try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

from pdfminer.high_level import extract_text as pdf_extract
from docx import Document

SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".md"}

# -------- Arabic heuristics --------
AR_RANGE = r"\u0600-\u06FF"

def _looks_ok_ar(text: str) -> bool:
    """تحقق سريع أن النص فعلاً عربي وليس مشوهاً."""
    if not text or len(text) < 80:
        return False
    ar_chars = re.findall(f"[{AR_RANGE}]", text)
    return (len(ar_chars) > 60) and ("�" not in text)

# -------- PDF extraction --------
def _extract_pdf_pymupdf(path: str) -> str:
    if fitz is None:
        return ""
    try:
        out = []
        with fitz.open(path) as doc:
            for page in doc:
                out.append(page.get_text("text"))
        return "\n".join(out)
    except Exception:
        return ""

def _extract_pdf_text(path: str) -> str:
    # 1) جرّب PyMuPDF أولاً
    text = _extract_pdf_pymupdf(path)
    if _looks_ok_ar(text):
        return text.strip()

    # 2) رجوع إلى pdfminer
    try:
        text2 = pdf_extract(str(path))
        if _looks_ok_ar(text2) or len(text2) > len(text):
            return (text2 or "").strip()
    except Exception:
        pass

    # 3) رجّع الأفضل المتاح (قد يكون قليل) — لاحقاً ممكن نضيف OCR
    return (text or "").strip()

# -------- Corpus inference --------
def _infer_corpus_from_path(p: Path) -> str:
    """
    استنتاج المصدر من المسار:
    - يحتوي على 'hr_policies' => 'hr'
    - يحتوي على 'jisr_guides' => 'jisr'
    - غير ذلك => 'unknown'
    """
    s = str(p).lower()
    if "hr_policies" in s:
        return "hr"
    if "jisr_guides" in s:
        return "jisr"
    return "unknown"

# -------- Main loader --------
def load_documents(folder: Path):
    folder = Path(folder)
    docs = []
    if not folder.exists():
        return docs

    corpus = _infer_corpus_from_path(folder)

    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTS:
            continue

        if ext == ".pdf":
            text = _extract_pdf_text(str(path))
        elif ext == ".docx":
            # جمع الفقرات من ملف وورد
            try:
                text = "\n".join(p.text for p in Document(path).paragraphs)
            except Exception:
                text = ""
        else:
            # txt/md
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = ""

        # ملاحظة: حتى لو الجودة ضعيفة، نُكمِل (يمكن لاحقاً نفعل OCR)
        # if not _looks_ok_ar(text): pass

        docs.append({
            "text": text,
            "meta": {
                "source": str(path),
                "doc_title": path.stem,
                "corpus": corpus,  # <-- مهم: نضيف الوسم هنا
            }
        })
    return docs
