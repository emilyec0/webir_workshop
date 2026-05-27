import pandas as pd
from pathlib import Path
import re
from typing import Optional
import hashlib
import json

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]

_DATA_CANDIDATES = [
    Path("data/book_details.csv"),  # expected export from data/book_details.db
    Path("data/Book_Details.csv"),  # provided workshop file
]

data_path = next((p for p in _DATA_CANDIDATES if p.exists()), None)
if data_path is None:
    raise FileNotFoundError(
        "No book dataset found. Expected one of: "
        + ", ".join(str(p) for p in _DATA_CANDIDATES)
    )

df = pd.read_csv(data_path)

df = df.fillna("")

df["search_text"] = (
    df.get("book_title", df.get("title", "")).astype(str)
    + " "
    + df.get("author", df.get("authors", "")).astype(str)
    + " "
    + df.get("book_details", df.get("description", "")).astype(str)
)

_MODEL_NAME = "all-MiniLM-L6-v2"
_CACHE_DIR = Path("data")
_EMBEDDINGS_CACHE = _CACHE_DIR / f".embeddings_{_MODEL_NAME}.npy"
_EMBEDDINGS_META = _CACHE_DIR / f".embeddings_{_MODEL_NAME}.json"
_MODEL: Optional["SentenceTransformer"] = None
_EMBEDDINGS: Optional[np.ndarray] = None


def _dataset_fingerprint(path: Path) -> str:
    st = path.stat()
    payload = f"{path.as_posix()}|{st.st_size}|{int(st.st_mtime)}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _load_or_build_embeddings(texts: list[str], dataset_path: Path) -> np.ndarray:
    if SentenceTransformer is None:
        raise RuntimeError(
            "sentence-transformers is not installed. Install requirements.txt to use all-MiniLM-L6-v2."
        )

    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(_MODEL_NAME)

    fingerprint = _dataset_fingerprint(dataset_path)
    if _EMBEDDINGS_CACHE.exists() and _EMBEDDINGS_META.exists():
        try:
            meta = json.loads(_EMBEDDINGS_META.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == fingerprint and meta.get("model") == _MODEL_NAME:
                emb = np.load(_EMBEDDINGS_CACHE)
                if emb.shape[0] == len(texts):
                    return emb
        except Exception:
            pass

    emb = _MODEL.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(_EMBEDDINGS_CACHE, emb)
    _EMBEDDINGS_META.write_text(
        json.dumps({"model": _MODEL_NAME, "fingerprint": fingerprint, "rows": len(texts)}, indent=2),
        encoding="utf-8",
    )
    return emb

_SIMILARITY_TRIGGERS = [
    r"\bsomething\s+like\s+(?P<ref>.+)$",
    r"\bbooks?\s+like\s+(?P<ref>.+)$",
    r"\blike\s+(?P<ref>.+)$",
    r"\bsimilar\s+to\s+(?P<ref>.+)$",
    r"\bsimilar\s+style\s+to\s+(?P<ref>.+)$",
    r"\bin\s+the\s+style\s+of\s+(?P<ref>.+)$",
]


def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\"'’`]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _extract_similarity_reference(query: str) -> Optional[str]:
    q = query.strip()
    q = re.sub(r"[?.!]+$", "", q).strip()
    for pattern in _SIMILARITY_TRIGGERS:
        m = re.search(pattern, q, flags=re.IGNORECASE)
        if not m:
            continue
        ref = m.group("ref").strip()
        ref = re.sub(r"^(a|an|the)\s+", "", ref, flags=re.IGNORECASE).strip()
        return ref if ref else None
    return None


def search_books(query, top_n=10):
    if not query.strip():
        return []

    similarity_ref = _extract_similarity_reference(query)
    normalized_ref = _normalize_text(similarity_ref) if similarity_ref else None

    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = _load_or_build_embeddings(df["search_text"].astype(str).tolist(), data_path)

    if SentenceTransformer is None:
        return []
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(_MODEL_NAME)

    query_emb = _MODEL.encode(
        [query], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
    )[0]

    scores = _EMBEDDINGS @ query_emb
    ranked_indices = np.argsort(scores)[::-1]

    results = []
    for i in ranked_indices:
        book = df.iloc[i]

        if normalized_ref:
            title = _normalize_text(str(book.get("book_title", book.get("title", ""))))
            author = _normalize_text(str(book.get("author", book.get("authors", ""))))

            # If the user asks for "something like X", don't return X itself.
            if normalized_ref in title or normalized_ref in author:
                continue

        results.append({
            "title": book.get("book_title", book.get("title", "Unknown Title")),
            "author": book.get("author", book.get("authors", "Unknown Author")),
            "rating": book.get("average_rating", "N/A"),
            "description": book.get("book_details", book.get("description", "No description available.")),
            "score": round(float(scores[i]), 3)
        })

        if len(results) >= top_n:
            break

    return results
