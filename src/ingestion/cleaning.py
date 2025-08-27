# src/ingestion/cleaning.py
import re
from typing import Dict

# Arabic combining marks (tashkeel/diacritics)
_AR_TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]")

# Tatweel (kashida)
_TATWEEL = "\u0640"

# Map of single-character fixes using ord() keys (required by str.translate)
# NOTE: Only single codepoints here.
_TRANSLATE_MAP = {
    ord("ى"): "ي",
    ord("ؤ"): "و",
    ord("ئ"): "ي",
    ord("إ"): "ا",
    ord("أ"): "ا",
    ord("آ"): "ا",
    ord("ٱ"): "ا",
}

# Regex for the 2-codepoint sequence: YEH + SUPERSCRIPT ALEF (U+064A U+0670)
_RE_YEH_SUPERSCRIPT_ALEF = re.compile("\u064A\u0670")

# Whitespace / punctuation normalization
_WS_MULTI = re.compile(r"[ \t\u00A0\u200f\u200e]+")
_NEWLINE_MULTI = re.compile(r"\n{3,}")
_DOT_SPACING = re.compile(r"\s*\.\s*")

def _normalize_arabic(text: str) -> str:
    if not text:
        return ""

    t = text

    # Remove tatweel
    t = t.replace(_TATWEEL, "")

    # Remove diacritics
    t = _AR_TASHKEEL.sub("", t)

    # Fix the two-codepoint sequence: "يٰ" → "ي"
    t = _RE_YEH_SUPERSCRIPT_ALEF.sub("ي", t)

    # Single-character translations
    t = t.translate(_TRANSLATE_MAP)

    # Normalize dot spacing
    t = _DOT_SPACING.sub(". ", t)

    # Collapse repeated whitespace
    t = _WS_MULTI.sub(" ", t)

    # Collapse many blank lines
    t = _NEWLINE_MULTI.sub("\n\n", t)

    return t.strip()

def clean_document(doc: Dict) -> Dict:
    """
    Input: {'text': str, 'meta': {...}}
    Output: normalized Arabic text (no change to meta)
    """
    text = (doc.get("text") or "")
    text = _normalize_arabic(text)
    meta = dict(doc.get("meta") or {})
    return {"text": text, "meta": meta}
