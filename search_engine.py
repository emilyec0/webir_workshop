import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/goodreads.csv")

df = df.fillna("")

df["search_text"] = (
    df["title"].astype(str) + " " +
    df["authors"].astype(str) + " " +
    df.get("description", "").astype(str)
)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["search_text"])

def search_books(query, top_n=10):
    if not query.strip():
        return []

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    top_indices = scores.argsort()[::-1][:top_n]

    results = []
    for i in top_indices:
        book = df.iloc[i]
        results.append({
            "title": book.get("title", "Unknown Title"),
            "author": book.get("authors", "Unknown Author"),
            "rating": book.get("average_rating", "N/A"),
            "description": book.get("description", "No description available."),
            "score": round(scores[i], 3)
        })

    return results