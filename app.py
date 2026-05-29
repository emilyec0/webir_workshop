import os

from flask import Flask, render_template, request
from search_engine import search_books, start_search_warmup

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search")
def search():
    query = request.args.get("q", "")
    sort_by = request.args.get("sort", "relevance")
    genre = request.args.get("genre", "")

    results = search_books(query, top_n=30, sort_by=sort_by, genre_filter=genre)

    return render_template(
        "results.html",
        query=query,
        results=results,
        sort_by=sort_by,
        genre=genre
    )


if __name__ == "__main__":
    debug = True

    if not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        start_search_warmup()

    app.run(debug=debug, use_reloader=False)
