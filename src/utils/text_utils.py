import re

ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")

def normalize_ar(text: str) -> str:
    text = ARABIC_DIACRITICS.sub("", text or "")
    text = text.replace("\u0640", "")
    text = text.replace("\u06CC", "\u064A").replace("\u0649", "\u064A")
    text = text.replace("\u06A9", "\u0643")
    return text.strip()
