import pandas as pd
from pathlib import Path
import threading
import re
import ast
from typing import Optional
import hashlib
import json
import sqlite3
from collections import defaultdict

import numpy as np

SentenceTransformer = None


def _get_sentence_transformer_class():
    global SentenceTransformer

    if SentenceTransformer is not None:
        return SentenceTransformer

    try:
        from sentence_transformers import SentenceTransformer as sentence_transformer_class
    except Exception:
        return None

    SentenceTransformer = sentence_transformer_class
    return SentenceTransformer

_DATA_CANDIDATES = [
    Path("data/book_details.csv"),
    Path("data/Book_Details.csv"),
]

data_path = next((p for p in _DATA_CANDIDATES if p.exists()), None)

if data_path is None:
    raise FileNotFoundError(
        "No book dataset found. Expected one of: "
        + ", ".join(str(p) for p in _DATA_CANDIDATES)
    )

_DATAFRAME: Optional[pd.DataFrame] = None
_TITLE_COL = ""
_AUTHOR_COL = ""
_DETAILS_COL = ""
_GENRE_COL = ""
_IMAGE_COL: Optional[str] = None
_SEARCH_TEXTS: list[str] = []
_BOOK_ID_VALUES = np.array([], dtype=np.int64)
_BOOK_ID_TO_INDEX: dict[int, int] = {}
_TITLE_VALUES = np.array([], dtype=str)
_AUTHOR_VALUES = np.array([], dtype=str)
_DETAILS_VALUES = np.array([], dtype=str)
_RATING_VALUES = np.array([], dtype=str)
_GENRE_VALUES = np.array([], dtype=str)
_IMAGE_VALUES = np.array([], dtype=str)
_TITLE_NORMALIZED = np.array([], dtype=object)
_AUTHOR_NORMALIZED = np.array([], dtype=object)
_GENRE_LOWER = np.array([], dtype=str)

_MODEL_NAME = "all-MiniLM-L6-v2"

_CACHE_DIR = Path("data")

_EMBEDDINGS_CACHE = _CACHE_DIR / f".embeddings_{_MODEL_NAME}.npy"
_EMBEDDINGS_META = _CACHE_DIR / f".embeddings_{_MODEL_NAME}.json"
_TOKEN_INDEX_CACHE = _CACHE_DIR / f".token_index_{_MODEL_NAME}.sqlite"
_TOKEN_INDEX_META = _CACHE_DIR / f".token_index_{_MODEL_NAME}.json"
_REVIEW_FEATURES_CACHE = _CACHE_DIR / f".review_features_{_MODEL_NAME}.npz"
_REVIEW_FEATURES_META = _CACHE_DIR / f".review_features_{_MODEL_NAME}.json"
_REVIEW_SENTIMENT_CACHE = _CACHE_DIR / f".review_sentiment_{_MODEL_NAME}.npz"
_REVIEW_SENTIMENT_META = _CACHE_DIR / f".review_sentiment_{_MODEL_NAME}.json"

_MODEL: Optional["SentenceTransformer"] = None
_EMBEDDINGS: Optional[np.ndarray] = None
_TOKEN_INDEX_CONNECTION: Optional[sqlite3.Connection] = None
_REVIEW_COUNTS = np.array([], dtype=np.int32)
_REVIEW_SCORES = np.array([], dtype=np.float32)
_REVIEW_AVG_RATINGS = np.array([], dtype=np.float32)
_REVIEW_AVG_LIKES = np.array([], dtype=np.float32)
_REVIEW_SENTIMENT_SCORES = np.array([], dtype=np.float32)
_REVIEW_SENTIMENT_COUNTS = np.array([], dtype=np.int32)
_INITIALIZATION_LOCK = threading.Lock()
_INITIALIZATION_THREAD_STARTED = False
_TOKEN_INDEX_BUILD_STARTED = False
_REVIEW_FEATURE_BUILD_STARTED = False
_REVIEW_SENTIMENT_BUILD_STARTED = False

_QUERY_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SEARCH_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "about",
    "like",
    "book",
    "books",
    "novel",
    "novels",
    "story",
    "stories",
}

_POSITIVE_SENTIMENT_WORDS = {
    "amazing",
    "awesome",
    "beautiful",
    "best",
    "captivating",
    "charming",
    "delightful",
    "enjoyable",
    "excellent",
    "exceptional",
    "favorite",
    "fun",
    "good",
    "great",
    "heartwarming",
    "impressive",
    "incredible",
    "love",
    "loved",
    "lovely",
    "masterpiece",
    "moving",
    "perfect",
    "powerful",
    "recommend",
    "wonderful",
    "well",
}

_NEGATIVE_SENTIMENT_WORDS = {
    "annoying",
    "bad",
    "boring",
    "confusing",
    "cringe",
    "dull",
    "forgettable",
    "frustrating",
    "hard",
    "hate",
    "hated",
    "messy",
    "negative",
    "poor",
    "predictable",
    "slow",
    "terrible",
    "tedious",
    "uninteresting",
    "weak",
    "worse",
    "worst",
}

_NEGATION_WORDS = {"not", "no", "never", "without", "hardly", "rarely"}


def _ensure_dataset_loaded() -> None:
    global _DATAFRAME
    global _TITLE_COL, _AUTHOR_COL, _DETAILS_COL, _GENRE_COL, _IMAGE_COL
    global _SEARCH_TEXTS, _TITLE_VALUES, _AUTHOR_VALUES, _DETAILS_VALUES
    global _RATING_VALUES, _GENRE_VALUES, _IMAGE_VALUES
    global _TITLE_NORMALIZED, _AUTHOR_NORMALIZED, _GENRE_LOWER
    global _BOOK_ID_VALUES, _BOOK_ID_TO_INDEX

    if _DATAFRAME is not None:
        return

    with _INITIALIZATION_LOCK:
        if _DATAFRAME is not None:
            return

        data_frame = pd.read_csv(data_path).fillna("")

        _TITLE_COL = "book_title" if "book_title" in data_frame.columns else "title"
        _AUTHOR_COL = "author" if "author" in data_frame.columns else "authors"
        _DETAILS_COL = (
            "book_details" if "book_details" in data_frame.columns else "description"
        )
        _GENRE_COL = (
            "genres"
            if "genres" in data_frame.columns
            else ("genre" if "genre" in data_frame.columns else "categories")
        )
        _IMAGE_COL = (
            "cover_image_uri"
            if "cover_image_uri" in data_frame.columns
            else (
                "image_url"
                if "image_url" in data_frame.columns
                else (
                    "thumbnail"
                    if "thumbnail" in data_frame.columns
                    else (
                        "cover_image"
                        if "cover_image" in data_frame.columns
                        else ("book_image" if "book_image" in data_frame.columns else None)
                    )
                )
            )
        )

        def column_to_values(column_name: Optional[str], default_value: str) -> np.ndarray:
            if column_name and column_name in data_frame.columns:
                return data_frame[column_name].astype(str).to_numpy()
            return np.array([default_value] * len(data_frame), dtype=str)

        title_values = column_to_values(_TITLE_COL, "Unknown Title")
        author_values = column_to_values(_AUTHOR_COL, "Unknown Author")
        details_values = column_to_values(_DETAILS_COL, "No description available.")
        rating_values = column_to_values("average_rating", "N/A")
        genre_values = column_to_values(_GENRE_COL, "")
        image_values = column_to_values(_IMAGE_COL, "") if _IMAGE_COL is not None else np.array([""] * len(data_frame), dtype=str)
        if "book_id" in data_frame.columns:
            book_id_values = pd.to_numeric(data_frame["book_id"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        else:
            book_id_values = np.arange(len(data_frame), dtype=np.int64)

        data_frame["search_text"] = (
            pd.Series(title_values)
            + " "
            + pd.Series(author_values)
            + " "
            + pd.Series(details_values)
        )

        _DATAFRAME = data_frame
        _SEARCH_TEXTS = data_frame["search_text"].astype(str).tolist()
        _TITLE_VALUES = title_values
        _AUTHOR_VALUES = author_values
        _DETAILS_VALUES = details_values
        _RATING_VALUES = rating_values
        _GENRE_VALUES = genre_values
        _IMAGE_VALUES = image_values
        _BOOK_ID_VALUES = book_id_values
        _BOOK_ID_TO_INDEX = {
            int(book_id): index
            for index, book_id in enumerate(book_id_values)
            if int(book_id) >= 0
        }
        _TITLE_NORMALIZED = np.array([None] * len(data_frame), dtype=object)
        _AUTHOR_NORMALIZED = np.array([None] * len(data_frame), dtype=object)
        _GENRE_LOWER = np.char.lower(genre_values.astype(str))


def _dataset_fingerprint(path: Path) -> str:
    st = path.stat()

    payload = (
        f"{path.as_posix()}|{st.st_size}|{int(st.st_mtime)}"
        .encode("utf-8")
    )

    return hashlib.sha1(payload).hexdigest()


def _load_or_build_embeddings(
    texts: list[str],
    dataset_path: Path
) -> np.ndarray:

    sentence_transformer_class = _get_sentence_transformer_class()

    if sentence_transformer_class is None:
        raise RuntimeError(
            "sentence-transformers is not installed."
        )

    global _MODEL

    if _MODEL is None:
        _MODEL = sentence_transformer_class(_MODEL_NAME)

    fingerprint = _dataset_fingerprint(dataset_path)

    if (
        _EMBEDDINGS_CACHE.exists()
        and _EMBEDDINGS_META.exists()
    ):
        try:
            meta = json.loads(
                _EMBEDDINGS_META.read_text(
                    encoding="utf-8"
                )
            )

            if (
                meta.get("fingerprint") == fingerprint
                and meta.get("model") == _MODEL_NAME
            ):
                emb = np.load(_EMBEDDINGS_CACHE, mmap_mode="r")

                if emb.shape[0] == len(texts):
                    return emb

        except Exception:
            pass

    emb = _MODEL.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    np.save(_EMBEDDINGS_CACHE, emb)

    _EMBEDDINGS_META.write_text(
        json.dumps(
            {
                "model": _MODEL_NAME,
                "fingerprint": fingerprint,
                "rows": len(texts)
            },
            indent=2
        ),
        encoding="utf-8",
    )

    return emb


def _load_token_index_connection() -> Optional[sqlite3.Connection]:
    global _TOKEN_INDEX_CONNECTION

    if _TOKEN_INDEX_CONNECTION is not None:
        return _TOKEN_INDEX_CONNECTION

    if not _TOKEN_INDEX_CACHE.exists() or not _TOKEN_INDEX_META.exists():
        return None

    try:
        meta = json.loads(_TOKEN_INDEX_META.read_text(encoding="utf-8"))
        if meta.get("fingerprint") != _dataset_fingerprint(data_path):
            return None

        connection = sqlite3.connect(
            f"file:{_TOKEN_INDEX_CACHE.as_posix()}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        connection.execute("PRAGMA query_only = ON")
        _TOKEN_INDEX_CONNECTION = connection
        return connection
    except Exception:
        return None


def _build_token_index() -> None:
    global _TOKEN_INDEX_CONNECTION

    _ensure_dataset_loaded()

    fingerprint = _dataset_fingerprint(data_path)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    existing_connection = _load_token_index_connection()
    if existing_connection is not None:
        _TOKEN_INDEX_CONNECTION = existing_connection
        return

    try:
        if _TOKEN_INDEX_CACHE.exists():
            _TOKEN_INDEX_CACHE.unlink()
        if _TOKEN_INDEX_META.exists():
            _TOKEN_INDEX_META.unlink()
    except Exception:
        pass

    connection = sqlite3.connect(
        _TOKEN_INDEX_CACHE.as_posix(),
        check_same_thread=False,
    )
    try:
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")
        connection.execute("PRAGMA temp_store = MEMORY")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS token_docs (
                token TEXT PRIMARY KEY,
                doc_ids BLOB NOT NULL,
                doc_freq INTEGER NOT NULL
            )
            """
        )

        token_to_docs = defaultdict(list)

        for doc_index, text in enumerate(_SEARCH_TEXTS):
            tokens = {
                token
                for token in _QUERY_TOKEN_RE.findall(text.lower())
                if len(token) > 2 and token not in _SEARCH_STOPWORDS
            }

            for token in tokens:
                token_to_docs[token].append(doc_index)

        rows = []
        for token, doc_ids in token_to_docs.items():
            doc_array = np.asarray(doc_ids, dtype=np.int32)
            rows.append((token, sqlite3.Binary(doc_array.tobytes()), int(len(doc_array))))

        connection.executemany(
            "INSERT INTO token_docs(token, doc_ids, doc_freq) VALUES (?, ?, ?)",
            rows,
        )
        connection.commit()

        _TOKEN_INDEX_META.write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "rows": len(_SEARCH_TEXTS),
                    "tokens": len(rows),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        connection.close()
        connection = sqlite3.connect(
            f"file:{_TOKEN_INDEX_CACHE.as_posix()}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        connection.execute("PRAGMA query_only = ON")
        _TOKEN_INDEX_CONNECTION = connection
        return
    except Exception:
        try:
            connection.close()
        except Exception:
            pass
        return


def _get_token_index_connection() -> Optional[sqlite3.Connection]:
    return _load_token_index_connection()


def _build_token_index_in_background() -> None:
    try:
        _build_token_index()
    except Exception:
        pass


def _parse_numeric_series(series: pd.Series) -> np.ndarray:
    extracted = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
    )
    return pd.to_numeric(extracted, errors="coerce").fillna(0).to_numpy()


def _review_sentiment_score(text: str) -> float:
    tokens = _QUERY_TOKEN_RE.findall(str(text).lower())
    if not tokens:
        return 0.0

    score = 0.0
    matched = 0
    negate_next = False

    for token in tokens:
        if token in _NEGATION_WORDS:
            negate_next = True
            continue

        token_score = 0.0
        if token in _POSITIVE_SENTIMENT_WORDS:
            token_score = 1.0
        elif token in _NEGATIVE_SENTIMENT_WORDS:
            token_score = -1.0

        if token_score == 0.0:
            continue

        matched += 1
        if negate_next:
            token_score *= -1.0
            negate_next = False

        score += token_score

    if matched == 0:
        return 0.0

    normalized = score / float(max(matched, 1))
    return float(np.clip(normalized, -1.0, 1.0))


def _load_review_features_cache() -> bool:
    global _REVIEW_COUNTS, _REVIEW_SCORES, _REVIEW_AVG_RATINGS, _REVIEW_AVG_LIKES

    if not _REVIEW_FEATURES_CACHE.exists() or not _REVIEW_FEATURES_META.exists():
        return False

    try:
        meta = json.loads(_REVIEW_FEATURES_META.read_text(encoding="utf-8"))
        if meta.get("fingerprint") != _dataset_fingerprint(Path("data/book_reviews.csv")):
            return False
        if meta.get("rows") != len(_BOOK_ID_VALUES):
            return False

        with np.load(_REVIEW_FEATURES_CACHE) as data:
            _REVIEW_COUNTS = data["counts"].astype(np.int32, copy=False)
            _REVIEW_SCORES = data["scores"].astype(np.float32, copy=False)
            _REVIEW_AVG_RATINGS = data["avg_ratings"].astype(np.float32, copy=False)
            _REVIEW_AVG_LIKES = data["avg_likes"].astype(np.float32, copy=False)

        return True
    except Exception:
        return False


def _build_review_features() -> None:
    global _REVIEW_COUNTS, _REVIEW_SCORES, _REVIEW_AVG_RATINGS, _REVIEW_AVG_LIKES

    _ensure_dataset_loaded()

    review_path = Path("data/book_reviews.csv")
    if not review_path.exists():
        _REVIEW_COUNTS = np.zeros(len(_BOOK_ID_VALUES), dtype=np.int32)
        _REVIEW_SCORES = np.zeros(len(_BOOK_ID_VALUES), dtype=np.float32)
        _REVIEW_AVG_RATINGS = np.zeros(len(_BOOK_ID_VALUES), dtype=np.float32)
        _REVIEW_AVG_LIKES = np.zeros(len(_BOOK_ID_VALUES), dtype=np.float32)
        return

    if _load_review_features_cache():
        return

    counts = np.zeros(len(_BOOK_ID_VALUES), dtype=np.int32)
    rating_sums = np.zeros(len(_BOOK_ID_VALUES), dtype=np.float32)
    likes_sums = np.zeros(len(_BOOK_ID_VALUES), dtype=np.float32)
    weighted_rating_sums = np.zeros(len(_BOOK_ID_VALUES), dtype=np.float32)
    weight_sums = np.zeros(len(_BOOK_ID_VALUES), dtype=np.float32)

    usecols = ["book_id", "likes_on_review", "review_rating"]

    try:
        for chunk in pd.read_csv(review_path, usecols=usecols, chunksize=50000):
            chunk["book_id"] = pd.to_numeric(chunk["book_id"], errors="coerce").fillna(-1).astype(np.int64)
            chunk["review_rating"] = pd.to_numeric(chunk["review_rating"], errors="coerce")
            chunk["likes_on_review"] = _parse_numeric_series(chunk["likes_on_review"]).astype(np.float32)

            chunk = chunk[chunk["book_id"].isin(_BOOK_ID_TO_INDEX)]
            if chunk.empty:
                continue

            for book_id, group in chunk.groupby("book_id"):
                index = _BOOK_ID_TO_INDEX.get(int(book_id))
                if index is None:
                    continue

                ratings = group["review_rating"].dropna().to_numpy(dtype=np.float32)
                likes = group["likes_on_review"].to_numpy(dtype=np.float32)

                if len(ratings) == 0:
                    continue

                count = int(len(ratings))
                counts[index] += count
                rating_sums[index] += float(ratings.sum())
                likes_sums[index] += float(likes.sum())

                weights = 1.0 + np.log1p(np.maximum(likes, 0.0))
                weighted_rating_sums[index] += float((ratings * weights).sum())
                weight_sums[index] += float(weights.sum())

        avg_ratings = np.divide(
            rating_sums,
            np.maximum(counts, 1),
            out=np.zeros_like(rating_sums),
            where=counts > 0,
        )
        avg_likes = np.divide(
            likes_sums,
            np.maximum(counts, 1),
            out=np.zeros_like(likes_sums),
            where=counts > 0,
        )
        weighted_avg_ratings = np.divide(
            weighted_rating_sums,
            np.maximum(weight_sums, 1e-6),
            out=np.zeros_like(weighted_rating_sums),
            where=weight_sums > 0,
        )

        count_norm = np.zeros_like(avg_ratings)
        likes_norm = np.zeros_like(avg_likes)
        if counts.max() > 0:
            count_norm = np.log1p(counts) / np.log1p(counts.max())
        if avg_likes.max() > 0:
            likes_norm = np.log1p(avg_likes) / np.log1p(avg_likes.max())

        review_scores = (
            0.7 * (weighted_avg_ratings / 5.0)
            + 0.2 * count_norm
            + 0.1 * likes_norm
        ).astype(np.float32)

        _REVIEW_COUNTS = counts
        _REVIEW_SCORES = review_scores
        _REVIEW_AVG_RATINGS = avg_ratings.astype(np.float32)
        _REVIEW_AVG_LIKES = avg_likes.astype(np.float32)

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(
            _REVIEW_FEATURES_CACHE,
            counts=_REVIEW_COUNTS,
            scores=_REVIEW_SCORES,
            avg_ratings=_REVIEW_AVG_RATINGS,
            avg_likes=_REVIEW_AVG_LIKES,
        )
        _REVIEW_FEATURES_META.write_text(
            json.dumps(
                {
                    "fingerprint": _dataset_fingerprint(review_path),
                    "rows": len(_BOOK_ID_VALUES),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        _REVIEW_COUNTS = np.zeros(len(_BOOK_ID_VALUES), dtype=np.int32)
        _REVIEW_SCORES = np.zeros(len(_BOOK_ID_VALUES), dtype=np.float32)
        _REVIEW_AVG_RATINGS = np.zeros(len(_BOOK_ID_VALUES), dtype=np.float32)
        _REVIEW_AVG_LIKES = np.zeros(len(_BOOK_ID_VALUES), dtype=np.float32)


def _ensure_review_features() -> None:
    global _REVIEW_FEATURE_BUILD_STARTED

    if _REVIEW_SCORES.size == len(_BOOK_ID_VALUES) and len(_BOOK_ID_VALUES) > 0:
        return

    if _load_review_features_cache():
        return

    if not _REVIEW_FEATURE_BUILD_STARTED:
        _REVIEW_FEATURE_BUILD_STARTED = True
        threading.Thread(
            target=_build_review_features,
            name="review-feature-build",
            daemon=True,
        ).start()


def _load_review_sentiment_cache() -> bool:
    global _REVIEW_SENTIMENT_SCORES, _REVIEW_SENTIMENT_COUNTS

    if not _REVIEW_SENTIMENT_CACHE.exists() or not _REVIEW_SENTIMENT_META.exists():
        return False

    try:
        meta = json.loads(_REVIEW_SENTIMENT_META.read_text(encoding="utf-8"))
        if meta.get("version") != 1:
            return False
        if meta.get("fingerprint") != _dataset_fingerprint(Path("data/book_reviews.csv")):
            return False
        if meta.get("rows") != len(_BOOK_ID_VALUES):
            return False

        with np.load(_REVIEW_SENTIMENT_CACHE) as data:
            _REVIEW_SENTIMENT_SCORES = data["sentiment_scores"].astype(np.float32, copy=False)
            _REVIEW_SENTIMENT_COUNTS = data["sentiment_counts"].astype(np.int32, copy=False)

        return True
    except Exception:
        return False


def _build_review_sentiment() -> None:
    global _REVIEW_SENTIMENT_SCORES, _REVIEW_SENTIMENT_COUNTS

    _ensure_dataset_loaded()

    review_path = Path("data/book_reviews.csv")
    if not review_path.exists():
        _REVIEW_SENTIMENT_SCORES = np.zeros(len(_BOOK_ID_VALUES), dtype=np.float32)
        _REVIEW_SENTIMENT_COUNTS = np.zeros(len(_BOOK_ID_VALUES), dtype=np.int32)
        return

    if _load_review_sentiment_cache():
        return

    sentiment_sums = np.zeros(len(_BOOK_ID_VALUES), dtype=np.float32)
    sentiment_weights = np.zeros(len(_BOOK_ID_VALUES), dtype=np.float32)
    sentiment_counts = np.zeros(len(_BOOK_ID_VALUES), dtype=np.int32)

    usecols = ["book_id", "review_content", "likes_on_review", "review_rating"]

    try:
        for chunk in pd.read_csv(review_path, usecols=usecols, chunksize=25000):
            chunk["book_id"] = pd.to_numeric(chunk["book_id"], errors="coerce").fillna(-1).astype(np.int64)
            chunk["likes_on_review"] = _parse_numeric_series(chunk["likes_on_review"]).astype(np.float32)
            chunk["review_rating"] = pd.to_numeric(chunk["review_rating"], errors="coerce").fillna(0).astype(np.float32)
            chunk["review_content"] = chunk["review_content"].astype(str)

            chunk = chunk[chunk["book_id"].isin(_BOOK_ID_TO_INDEX)]
            if chunk.empty:
                continue

            for book_id, group in chunk.groupby("book_id"):
                index = _BOOK_ID_TO_INDEX.get(int(book_id))
                if index is None:
                    continue

                sentiment_values = group["review_content"].map(_review_sentiment_score).to_numpy(dtype=np.float32)
                if len(sentiment_values) == 0:
                    continue

                likes = group["likes_on_review"].to_numpy(dtype=np.float32)
                ratings = group["review_rating"].to_numpy(dtype=np.float32)

                # More helpful reviews weigh a bit more, but text sentiment stays the core signal.
                weights = 1.0 + np.log1p(np.maximum(likes, 0.0))
                weights *= 1.0 + np.maximum(ratings - 3.0, 0.0) * 0.05

                sentiment_sums[index] += float((sentiment_values * weights).sum())
                sentiment_weights[index] += float(weights.sum())
                sentiment_counts[index] += int(len(sentiment_values))

        sentiment_scores = np.divide(
            sentiment_sums,
            np.maximum(sentiment_weights, 1e-6),
            out=np.zeros_like(sentiment_sums),
            where=sentiment_weights > 0,
        )

        _REVIEW_SENTIMENT_SCORES = np.clip(sentiment_scores, -1.0, 1.0).astype(np.float32)
        _REVIEW_SENTIMENT_COUNTS = sentiment_counts

        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(
            _REVIEW_SENTIMENT_CACHE,
            sentiment_scores=_REVIEW_SENTIMENT_SCORES,
            sentiment_counts=_REVIEW_SENTIMENT_COUNTS,
        )
        _REVIEW_SENTIMENT_META.write_text(
            json.dumps(
                {
                    "version": 1,
                    "fingerprint": _dataset_fingerprint(review_path),
                    "rows": len(_BOOK_ID_VALUES),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        _REVIEW_SENTIMENT_SCORES = np.zeros(len(_BOOK_ID_VALUES), dtype=np.float32)
        _REVIEW_SENTIMENT_COUNTS = np.zeros(len(_BOOK_ID_VALUES), dtype=np.int32)


def _ensure_review_sentiment() -> None:
    global _REVIEW_SENTIMENT_BUILD_STARTED

    if _REVIEW_SENTIMENT_SCORES.size == len(_BOOK_ID_VALUES) and len(_BOOK_ID_VALUES) > 0:
        return

    if _load_review_sentiment_cache():
        return

    if not _REVIEW_SENTIMENT_BUILD_STARTED:
        _REVIEW_SENTIMENT_BUILD_STARTED = True
        threading.Thread(
            target=_build_review_sentiment,
            name="review-sentiment-build",
            daemon=True,
        ).start()


def build_review_sentiment_cache() -> None:
    _build_review_sentiment()


def _review_boost_for_indices(indices: np.ndarray) -> np.ndarray:
    if _REVIEW_SCORES.size == 0 or len(indices) == 0:
        return np.zeros(len(indices), dtype=np.float32)

    return _REVIEW_SCORES[np.asarray(indices, dtype=np.int64)]


def _review_sentiment_for_indices(indices: np.ndarray) -> np.ndarray:
    if _REVIEW_SENTIMENT_SCORES.size == 0 or len(indices) == 0:
        return np.zeros(len(indices), dtype=np.float32)

    return _REVIEW_SENTIMENT_SCORES[np.asarray(indices, dtype=np.int64)]


def _field_match_strength(
    query_norm: str,
    query_tokens: set[str],
    field_norm: str,
) -> float:
    if not query_norm or not field_norm:
        return 0.0

    if query_norm == field_norm:
        return 1.0

    if query_norm in field_norm or field_norm in query_norm:
        return 0.92 if len(query_tokens) <= 3 else 0.86

    if not query_tokens:
        return 0.0

    field_tokens = set(_QUERY_TOKEN_RE.findall(field_norm))
    if not field_tokens:
        return 0.0

    overlap = len(query_tokens & field_tokens)
    if overlap == 0:
        return 0.0

    query_coverage = overlap / float(len(query_tokens))
    field_coverage = overlap / float(len(field_tokens))
    return min(0.8, 0.52 * query_coverage + 0.28 * field_coverage + 0.12)


def _field_boost_for_indices(
    query: str,
    indices: np.ndarray,
    normalized_ref: Optional[str],
) -> np.ndarray:
    if len(indices) == 0:
        return np.array([], dtype=np.float32)

    query_norm = _normalize_text(query)
    query_tokens = {
        token
        for token in _QUERY_TOKEN_RE.findall(query_norm)
        if len(token) > 2 and token not in _SEARCH_STOPWORDS
    }

    ref_norm = normalized_ref or ""
    ref_tokens = {
        token
        for token in _QUERY_TOKEN_RE.findall(ref_norm)
        if len(token) > 2 and token not in _SEARCH_STOPWORDS
    }

    boosts = np.zeros(len(indices), dtype=np.float32)

    for position, index in enumerate(np.asarray(indices, dtype=np.int64)):
        title_norm, author_norm = _normalized_title_and_author(int(index))

        title_score = max(
            _field_match_strength(query_norm, query_tokens, title_norm),
            _field_match_strength(ref_norm, ref_tokens, title_norm),
        )
        author_score = max(
            _field_match_strength(query_norm, query_tokens, author_norm),
            _field_match_strength(ref_norm, ref_tokens, author_norm),
        )

        if len(query_tokens) <= 3:
            author_score *= 1.08

        boosts[position] = max(title_score, author_score)

    return boosts


def _blend_relevance_with_reviews(
    base_scores: np.ndarray,
    indices: np.ndarray,
    query: str = "",
    normalized_ref: Optional[str] = None,
) -> np.ndarray:

    if len(indices) == 0:
        return np.array([], dtype=np.float32)

    review_scores = _review_boost_for_indices(indices)
    sentiment_scores = _review_sentiment_for_indices(indices)
    field_boosts = _field_boost_for_indices(query, indices, normalized_ref)

    if review_scores.size == 0 and sentiment_scores.size == 0 and field_boosts.size == 0:
        return np.asarray(base_scores, dtype=np.float32)

    base_scores = np.asarray(base_scores, dtype=np.float32)
    if base_scores.size == 0:
        base_scores = np.zeros(len(indices), dtype=np.float32)

    base_min = float(base_scores.min())
    base_max = float(base_scores.max())
    if base_max > base_min:
        base_norm = (base_scores - base_min) / (base_max - base_min)
    else:
        base_norm = np.ones_like(base_scores, dtype=np.float32)

    review_min = float(review_scores.min()) if review_scores.size else 0.0
    review_max = float(review_scores.max()) if review_scores.size else 0.0
    if review_max > review_min:
        review_norm = (review_scores - review_min) / (review_max - review_min)
    else:
        review_norm = np.zeros_like(review_scores, dtype=np.float32)

    sentiment_norm = (np.clip(sentiment_scores, -1.0, 1.0) + 1.0) / 2.0

    return 0.58 * base_norm + 0.14 * review_norm + 0.08 * sentiment_norm + 0.20 * field_boosts


def _initialize_search_assets() -> None:
    global _EMBEDDINGS, _MODEL

    if _get_sentence_transformer_class() is None:
        return

    _ensure_dataset_loaded()

    if _EMBEDDINGS is not None:
        return

    with _INITIALIZATION_LOCK:
        if _EMBEDDINGS is not None:
            return

        _EMBEDDINGS = _load_or_build_embeddings(_SEARCH_TEXTS, data_path)

    if _MODEL is not None:
        _MODEL.encode(
            ["warmup"],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    if _EMBEDDINGS is not None and len(_EMBEDDINGS) > 0:
        _ = float(_EMBEDDINGS[0] @ _EMBEDDINGS[0])


def start_search_warmup() -> None:
    global _INITIALIZATION_THREAD_STARTED, _TOKEN_INDEX_BUILD_STARTED, _REVIEW_FEATURE_BUILD_STARTED, _REVIEW_SENTIMENT_BUILD_STARTED

    if _INITIALIZATION_THREAD_STARTED:
        return

    _INITIALIZATION_THREAD_STARTED = True

    if not _TOKEN_INDEX_BUILD_STARTED:
        if _load_token_index_connection() is None:
            _TOKEN_INDEX_BUILD_STARTED = True
            threading.Thread(
                target=_build_token_index_in_background,
                name="token-index-build",
                daemon=True,
            ).start()

    if not _REVIEW_FEATURE_BUILD_STARTED and not _load_review_features_cache():
        _REVIEW_FEATURE_BUILD_STARTED = True
        threading.Thread(
            target=_build_review_features,
            name="review-feature-build",
            daemon=True,
        ).start()

    if not _REVIEW_SENTIMENT_BUILD_STARTED and not _load_review_sentiment_cache():
        _REVIEW_SENTIMENT_BUILD_STARTED = True
        threading.Thread(
            target=_build_review_sentiment,
            name="review-sentiment-build",
            daemon=True,
        ).start()

    threading.Thread(
        target=_initialize_search_assets,
        name="search-engine-warmup",
        daemon=True,
    ).start()


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


def _extract_similarity_reference(
    query: str
) -> Optional[str]:

    q = query.strip()

    q = re.sub(r"[?.!]+$", "", q).strip()

    for pattern in _SIMILARITY_TRIGGERS:
        m = re.search(
            pattern,
            q,
            flags=re.IGNORECASE
        )

        if not m:
            continue

        ref = m.group("ref").strip()

        ref = re.sub(
            r"^(a|an|the)\s+",
            "",
            ref,
            flags=re.IGNORECASE
        ).strip()

        return ref if ref else None

    return None


def _rating_to_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def _parse_genre_tags(value) -> list[str]:
    if value is None:
        return []

    text = str(value).strip()
    if not text:
        return []

    parsed = None
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None

    if isinstance(parsed, (list, tuple)):
        tags = [str(item).strip().strip("'\"") for item in parsed]
    else:
        tags = [
            part.strip().strip("'\"")
            for part in re.split(r"[,\|/]+", text)
        ]

    cleaned = []
    seen = set()
    for tag in tags:
        if not tag:
            continue
        normalized_tag = re.sub(r"\s+", " ", tag).strip()
        lower_tag = normalized_tag.lower()
        if lower_tag in seen:
            continue
        seen.add(lower_tag)
        cleaned.append(normalized_tag)

    return cleaned


def _normalized_title_and_author(index: int) -> tuple[str, str]:
    title = _TITLE_NORMALIZED[index]
    if title is None:
        title = _TITLE_NORMALIZED[index] = _normalize_text(str(_TITLE_VALUES[index]))

    author = _AUTHOR_NORMALIZED[index]
    if author is None:
        author = _AUTHOR_NORMALIZED[index] = _normalize_text(str(_AUTHOR_VALUES[index]))

    return title, author


def _query_candidate_indices(query: str, top_n: int) -> Optional[np.ndarray]:
    connection = _get_token_index_connection()
    if connection is None:
        return None

    query_tokens = [
        token
        for token in _QUERY_TOKEN_RE.findall(query.lower())
        if len(token) > 2 and token not in _SEARCH_STOPWORDS
    ]

    if not query_tokens:
        return None

    candidate_limit = max(400, top_n * 80)
    candidate_docs: set[int] = set()

    placeholders = ",".join(["?"] * len(set(query_tokens)))
    rows = connection.execute(
        f"""
        SELECT token, doc_ids, doc_freq
        FROM token_docs
        WHERE token IN ({placeholders})
        ORDER BY doc_freq ASC
        """,
        tuple(set(query_tokens)),
    ).fetchall()

    for _, doc_blob, _ in rows:
        doc_ids = np.frombuffer(doc_blob, dtype=np.int32)
        candidate_docs.update(int(doc_id) for doc_id in doc_ids)
        if len(candidate_docs) >= candidate_limit:
            break

    if not candidate_docs:
        return None

    return np.fromiter(candidate_docs, dtype=np.int32)


def _rank_lexical_results(
    query: str,
    top_n: int,
    sort_by: str,
    genre_filter: str,
    normalized_ref: Optional[str],
) -> list[dict]:

    _ensure_dataset_loaded()
    _ensure_review_features()
    _ensure_review_sentiment()

    connection = _get_token_index_connection()
    if connection is None:
        return _rank_direct_lexical_results(
            query=query,
            top_n=top_n,
            sort_by=sort_by,
            genre_filter=genre_filter,
            normalized_ref=normalized_ref,
        )

    query_tokens = [
        token
        for token in _QUERY_TOKEN_RE.findall(query.lower())
        if len(token) > 2 and token not in _SEARCH_STOPWORDS
    ]

    if not query_tokens:
        return []

    unique_tokens = list(dict.fromkeys(query_tokens))
    placeholders = ",".join(["?"] * len(unique_tokens))
    rows = connection.execute(
        f"""
        SELECT token, doc_ids, doc_freq
        FROM token_docs
        WHERE token IN ({placeholders})
        ORDER BY doc_freq ASC
        """,
        tuple(unique_tokens),
    ).fetchall()

    if not rows:
        return []

    candidate_limit = max(400, top_n * 80)
    candidate_scores: dict[int, float] = defaultdict(float)
    genre_filter_lower = genre_filter.lower().strip()

    for _, doc_blob, doc_freq in rows:
        weight = 1.0 / (1.0 + float(doc_freq))
        doc_ids = np.frombuffer(doc_blob, dtype=np.int32)

        for doc_id in doc_ids:
            doc_index = int(doc_id)
            candidate_scores[doc_index] += weight

        if len(candidate_scores) >= candidate_limit:
            break

    if not candidate_scores:
        return []

    ranked_indices = sorted(
        candidate_scores.keys(),
        key=lambda index: candidate_scores[index],
        reverse=True,
    )

    ranked_base_scores = np.asarray(
        [candidate_scores[index] for index in ranked_indices],
        dtype=np.float32,
    )
    ranked_review_scores = _blend_relevance_with_reviews(
        ranked_base_scores,
        np.asarray(ranked_indices, dtype=np.int32),
        query=query,
        normalized_ref=normalized_ref,
    )

    results = []
    for position, index in enumerate(ranked_indices):
        if normalized_ref:
            title_norm, author_norm = _normalized_title_and_author(index)
            if normalized_ref in title_norm or normalized_ref in author_norm:
                continue

        if genre_filter_lower and genre_filter_lower not in _GENRE_LOWER[index]:
            continue

        genre_text = _GENRE_VALUES[index]
        genre_tags = _parse_genre_tags(genre_text)
        results.append({
            "title": _TITLE_VALUES[index] or "Unknown Title",
            "author": _AUTHOR_VALUES[index] or "Unknown Author",
            "rating": _RATING_VALUES[index],
            "genre": genre_tags[0] if genre_tags else (genre_text if genre_text else "Unlisted genre"),
            "tags": genre_tags,
            "description": _DETAILS_VALUES[index] or "No description available.",
            "image": _IMAGE_VALUES[index],
            "score": round(float(ranked_review_scores[position]), 3),
        })

        if len(results) >= max(top_n, 100):
            break

    if sort_by == "rating":
        results = sorted(results, key=lambda book: _rating_to_float(book["rating"]), reverse=True)
    elif sort_by == "title":
        results = sorted(results, key=lambda book: str(book["title"]).lower())
    elif sort_by == "author":
        results = sorted(results, key=lambda book: str(book["author"]).lower())

    return results[:top_n]


def _rank_direct_lexical_results(
    query: str,
    top_n: int,
    sort_by: str,
    genre_filter: str,
    normalized_ref: Optional[str],
) -> list[dict]:

    _ensure_dataset_loaded()
    _ensure_review_features()
    _ensure_review_sentiment()

    query_tokens = [
        token
        for token in _QUERY_TOKEN_RE.findall(query.lower())
        if len(token) > 2 and token not in _SEARCH_STOPWORDS
    ]

    if not query_tokens:
        return []

    unique_tokens = list(dict.fromkeys(query_tokens))
    genre_filter_lower = genre_filter.lower().strip()

    candidate_scores: list[tuple[float, int]] = []
    max_candidates = max(100, top_n)

    for index, text in enumerate(_SEARCH_TEXTS):
        lowered_text = text.lower()
        score = 0.0

        for token in unique_tokens:
            if token in lowered_text:
                score += 1.0 / (1.0 + len(token))

        if score == 0.0:
            continue

        if normalized_ref:
            title_norm, author_norm = _normalized_title_and_author(index)
            if normalized_ref in title_norm or normalized_ref in author_norm:
                continue

        if genre_filter_lower and genre_filter_lower not in _GENRE_LOWER[index]:
            continue

        candidate_scores.append((score, index))

    if not candidate_scores:
        return []

    base_scores = np.asarray([score for score, _ in candidate_scores], dtype=np.float32)
    indices = np.asarray([index for _, index in candidate_scores], dtype=np.int32)
    blended_scores = _blend_relevance_with_reviews(
        base_scores,
        indices,
        query=query,
        normalized_ref=normalized_ref,
    )
    candidate_scores = list(zip(blended_scores.tolist(), indices.tolist()))
    candidate_scores.sort(key=lambda item: item[0], reverse=True)

    results = []
    for score, index in candidate_scores[:max_candidates]:
        genre_text = _GENRE_VALUES[index]
        genre_tags = _parse_genre_tags(genre_text)
        results.append({
            "title": _TITLE_VALUES[index] or "Unknown Title",
            "author": _AUTHOR_VALUES[index] or "Unknown Author",
            "rating": _RATING_VALUES[index],
            "genre": genre_tags[0] if genre_tags else (genre_text if genre_text else "Unlisted genre"),
            "tags": genre_tags,
            "description": _DETAILS_VALUES[index] or "No description available.",
            "image": _IMAGE_VALUES[index],
            "score": round(score, 3),
        })

    if sort_by == "rating":
        results = sorted(results, key=lambda book: _rating_to_float(book["rating"]), reverse=True)
    elif sort_by == "title":
        results = sorted(results, key=lambda book: str(book["title"]).lower())
    elif sort_by == "author":
        results = sorted(results, key=lambda book: str(book["author"]).lower())

    return results[:top_n]


def _select_candidate_indices(
    scores: np.ndarray,
    target_count: int,
    genre_filter: str,
    normalized_ref: Optional[str],
    candidate_indices: Optional[np.ndarray] = None,
) -> np.ndarray:

    if candidate_indices is None:
        working_indices = np.arange(len(scores), dtype=np.int32)
        working_scores = scores
    else:
        working_indices = candidate_indices
        working_scores = scores

    total = len(working_indices)
    if total == 0:
        return np.array([], dtype=int)

    if target_count <= 0:
        target_count = 1

    candidate_pool = min(total, max(200, target_count * 20))

    genre_filter_lower = genre_filter.lower().strip()

    while True:
        if candidate_pool >= total:
            current_positions = np.arange(total, dtype=np.int32)
        else:
            current_positions = np.argpartition(working_scores, -candidate_pool)[-candidate_pool:]

        candidate_scores = working_scores[current_positions]
        order = np.argsort(candidate_scores)[::-1]
        ordered_indices = working_indices[current_positions[order]]

        if not genre_filter_lower and not normalized_ref:
            return ordered_indices[:target_count]

        filtered_indices = []
        for index in ordered_indices:
            if normalized_ref:
                title_norm, author_norm = _normalized_title_and_author(index)
                if normalized_ref in title_norm or normalized_ref in author_norm:
                    continue

            if genre_filter_lower and genre_filter_lower not in _GENRE_LOWER[index]:
                continue

            filtered_indices.append(index)
            if len(filtered_indices) >= target_count:
                return np.array(filtered_indices, dtype=int)

        if candidate_pool >= total:
            return np.array(filtered_indices, dtype=int)

        candidate_pool = min(total, candidate_pool * 2)


def search_books(query, top_n=10, sort_by="relevance", genre_filter=""):
    global _EMBEDDINGS
    global _MODEL

    if not query.strip():
        return []

    _ensure_dataset_loaded()
    _ensure_review_features()
    _ensure_review_sentiment()

    similarity_ref = _extract_similarity_reference(query)
    normalized_ref = _normalize_text(similarity_ref) if similarity_ref else None

    if _EMBEDDINGS is None or _MODEL is None:
        lexical_results = _rank_lexical_results(
            query=query,
            top_n=top_n,
            sort_by=sort_by,
            genre_filter=genre_filter,
            normalized_ref=normalized_ref,
        )
        if lexical_results:
            return lexical_results

        return _rank_direct_lexical_results(
            query=query,
            top_n=top_n,
            sort_by=sort_by,
            genre_filter=genre_filter,
            normalized_ref=normalized_ref,
        )

    sentence_transformer_class = _get_sentence_transformer_class()
    if sentence_transformer_class is None:
        return _rank_direct_lexical_results(
            query=query,
            top_n=top_n,
            sort_by=sort_by,
            genre_filter=genre_filter,
            normalized_ref=normalized_ref,
        )

    if _MODEL is None:
        _MODEL = sentence_transformer_class(_MODEL_NAME)

    query_emb = _MODEL.encode(
        [query], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True
    )[0]

    query_candidate_indices = _query_candidate_indices(query, top_n)

    if query_candidate_indices is None or len(query_candidate_indices) < top_n:
        query_candidate_indices = None
        scores = _EMBEDDINGS @ query_emb
    else:
        scores = _EMBEDDINGS[query_candidate_indices] @ query_emb

    selected_indices = _select_candidate_indices(
        scores=scores,
        target_count=max(top_n, 100),
        genre_filter=genre_filter,
        normalized_ref=normalized_ref,
        candidate_indices=query_candidate_indices,
    )

    if len(selected_indices) > 0:
        if query_candidate_indices is None:
            base_scores = np.asarray([float(scores[i]) for i in selected_indices], dtype=np.float32)
        else:
            selected_positions = {int(index): position for position, index in enumerate(query_candidate_indices)}
            base_scores = np.asarray(
                [float(scores[selected_positions[int(index)]]) for index in selected_indices],
                dtype=np.float32,
            )
        blended_scores = _blend_relevance_with_reviews(
            base_scores,
            selected_indices,
            query=query,
            normalized_ref=normalized_ref,
        )
    else:
        blended_scores = np.asarray([], dtype=np.float32)

    results = []

    for position, i in enumerate(selected_indices):
        genre_text = _GENRE_VALUES[i]
        genre_tags = _parse_genre_tags(genre_text)
        score_value = float(blended_scores[position]) if len(blended_scores) > position else 0.0

        results.append({
            "title": _TITLE_VALUES[i] or "Unknown Title",
            "author": _AUTHOR_VALUES[i] or "Unknown Author",
            "rating": _RATING_VALUES[i],
            "genre": genre_tags[0] if genre_tags else (genre_text if genre_text else "Unlisted genre"),
            "tags": genre_tags,
            "description": _DETAILS_VALUES[i] or "No description available.",
            "image": _IMAGE_VALUES[i],
            "score": round(score_value, 3)
        })

    if sort_by == "rating":
        results = sorted(results, key=lambda book: _rating_to_float(book["rating"]), reverse=True)
    elif sort_by == "title":
        results = sorted(results, key=lambda book: str(book["title"]).lower())
    elif sort_by == "author":
        results = sorted(results, key=lambda book: str(book["author"]).lower())

    return results[:top_n]
