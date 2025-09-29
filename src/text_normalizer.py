import re, unicodedata

def normalize(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8").lower()
    t = re.sub(r"[^a-z0-9#+ ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()
