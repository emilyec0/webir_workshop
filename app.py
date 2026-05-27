from flask import Flask, render_template, request
from search_engine import search_books

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search")
def search():
    query = request.args.get("q", "")
    results = search_books(query)
    return render_template("results.html", query=query, results=results)

if __name__ == "__main__":
    app.run(debug=True)