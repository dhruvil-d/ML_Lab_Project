# ============================================================
# app.py — Flask Backend for CineNLP / Cinematrix Dashboard
# Serves the web UI and exposes REST API endpoints for:
#   - Semantic movie search (cosine similarity on spaCy vectors)
#   - Movie detail lookup from the local CSV dataset
#   - AI genre classification (NB / LR / SVM on TF-IDF features)
#   - Word-to-plot relevance scoring (spaCy vector similarity)
#   - Similar movie recommendations (plot embedding similarity)
#   - Model performance metrics & EDA data for the dashboard
# ============================================================

# --- Standard library imports ---
import os        # For constructing file paths (os.path.join)
import re        # For regex-based text cleaning in preprocess()
import json      # For loading JSON config/metric files from disk
import pickle    # For deserialising trained sklearn models (.pkl files)

# --- Third-party imports ---
import numpy as np  # Numerical operations: dot products, norms for cosine similarity
import spacy        # NLP library for word vectors and semantic similarity

# --- Flask imports ---
from flask import Flask, request, jsonify, render_template
# Flask      — creates the web application instance
# request    — access incoming HTTP request data (query params, JSON body)
# jsonify    — convert Python dicts/lists to JSON HTTP responses
# render_template — render Jinja2 HTML templates from the templates/ folder

# Initialise the Flask application; __name__ tells Flask where to find
# templates/ and static/ directories relative to this file.
app = Flask(__name__)

# ============================================================
# STEP 1: Load the movie dataset from CSV at server startup
# ============================================================

# In-memory list that holds every movie as a dict (loaded from CSV below)
MOVIES_DB = []

# Resolve absolute paths to the models/ and data/ folders
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

import csv  # CSV reader to parse movies.csv

try:
    print("Loading movies dataset from CSV...")

    # Build the full path to data/movies.csv
    data_path = os.path.join(DATA_DIR, "movies.csv")

    # Open and iterate through each row using DictReader
    # (each row becomes a dict keyed by the CSV header columns)
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # The genres_list column was stored as pipe-delimited string
            # ("Action|Drama|Thriller") in the CSV.  Split it back into
            # a Python list for programmatic access.
            if "genres_list" in row and isinstance(row["genres_list"], str):
                row["genres_list"] = row["genres_list"].split("|")
            MOVIES_DB.append(row)

    print(f"✅ Loaded {len(MOVIES_DB)} movies from CSV.")
except Exception as e:
    print(f"⚠️ Error loading CSV: {e}")

# ============================================================
# STEP 2: Load pre-trained ML models and artifacts at startup
# ============================================================

def load_models():
    """
    Loads all pickled sklearn models, the TF-IDF vectorizer,
    label encoder, evaluation metrics, genre classes, movie
    embeddings, and EDA metrics from the models/ directory.
    Returns a dict mapping short keys to loaded objects.
    """
    models = {}
    try:
        # --- Trained classifiers (pickled sklearn estimators) ---

        # Naive Bayes classifier for genre prediction
        with open(os.path.join(MODELS_DIR, "nb_model.pkl"), "rb") as f:
            models["nb"] = pickle.load(f)

        # Logistic Regression classifier for genre prediction
        with open(os.path.join(MODELS_DIR, "lr_model.pkl"), "rb") as f:
            models["lr"] = pickle.load(f)

        # SVM classifier (CalibratedClassifierCV wrapper for probabilities)
        with open(os.path.join(MODELS_DIR, "svm_model.pkl"), "rb") as f:
            models["svm"] = pickle.load(f)

        # --- Feature engineering artifacts ---

        # TF-IDF vectorizer fitted on training plots; transforms raw text → sparse matrix
        with open(os.path.join(MODELS_DIR, "vectorizer.pkl"), "rb") as f:
            models["vectorizer"] = pickle.load(f)

        # Label encoder that maps genre strings ↔ integer class indices
        with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "rb") as f:
            models["label_encoder"] = pickle.load(f)

        # --- Evaluation & metadata ---

        # Dict of accuracy/precision/recall/f1/confusion matrices per model
        with open(os.path.join(MODELS_DIR, "metrics.json"), "r") as f:
            models["metrics"] = json.load(f)

        # Ordered list of genre class names (matches label encoder classes)
        with open(os.path.join(MODELS_DIR, "genre_classes.json"), "r") as f:
            models["genre_classes"] = json.load(f)

        # --- Semantic search & recommendation artifacts ---

        # Dict mapping imdbID → numpy vector (spaCy plot embeddings)
        # Used for semantic search and similar-movie recommendations
        with open(os.path.join(MODELS_DIR, "movie_embeddings.pkl"), "rb") as f:
            models["movie_embeddings"] = pickle.load(f)

        # --- EDA (Exploratory Data Analysis) metrics ---

        # Genre distribution counts and top TF-IDF words per genre
        with open(os.path.join(MODELS_DIR, "eda_metrics.json"), "r") as f:
            models["eda_metrics"] = json.load(f)

        print("✅ All models, embeddings, and EDA metrics loaded successfully!")

    except FileNotFoundError as e:
        print(f"⚠️  Model files not found: {e}")
        print("   Run fetch_data.py then train_models.py first.")

    return models

# Execute model loading immediately on import / server start
MODELS = load_models()

# ============================================================
# STEP 3: Load the spaCy medium English model for word vectors
# ============================================================
# en_core_web_md provides 300-dimensional word vectors which are
# used for:  (a) semantic search, (b) word relevance scoring
try:
    print("Loading spaCy model...")
    NLP = spacy.load("en_core_web_md")
    print("✅ spaCy model loaded!")
except OSError:
    # Graceful degradation — search will fall back to keyword matching
    print("⚠️  spaCy model en_core_web_md not found.")
    NLP = None


# ============================================================
# STEP 4: Text preprocessing helper
# ============================================================

def preprocess(text):
    """
    Clean raw text for TF-IDF vectorization:
      1. Lowercase the entire string for case-insensitive matching
      2. Strip all non-alphabetic characters (numbers, punctuation)
    Returns the cleaned string.
    """
    text = text.lower()                        # Normalise case
    text = re.sub(r'[^a-zA-Z\s]', '', text)    # Keep only letters & whitespace
    return text


# ============================================================
# ROUTES — Flask API Endpoints
# ============================================================

# ---------- Home page ----------
@app.route("/")
def index():
    """Serve the single-page HTML frontend (templates/index.html)."""
    return render_template("index.html")


# ---------- Movie search ----------
@app.route("/api/search")
def search_movies():
    """
    Search movies using Semantic Search (cosine similarity on spaCy
    plot embeddings) with an automatic fallback to keyword matching
    if embeddings or the spaCy model aren't available.

    Query params:
        q    — search query string (required)
        page — page number for pagination (default 1, 10 results/page)

    Returns JSON:
        { movies: [...], totalResults: int, source: "semantic"|"keyword_fallback" }
    """
    # Read and validate the search query from URL parameters
    query = request.args.get("q", "").strip()
    page = request.args.get("page", "1")

    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Safely parse the page number; default to 1 on invalid input
    try:
        page_num = int(page)
    except ValueError:
        page_num = 1

    q = query.lower()  # Lowercase query for case-insensitive comparisons
    matched = []        # Will hold the final list of matched movie dicts

    # --- Primary path: Semantic similarity search ---
    # Requires pre-computed movie embeddings AND a loaded spaCy model
    embeddings = MODELS.get("movie_embeddings")
    if embeddings and NLP:
        # Convert the user's query into a 300-dim spaCy vector
        query_vec = NLP(query).vector
        query_norm = np.linalg.norm(query_vec)  # L2 norm for cosine sim denominator

        scored_movies = []  # List of (score, movie_dict) tuples
        for m in MOVIES_DB:
            score = 0

            # 1. Exact-match boost:  if query appears in title or genre,
            #    give a large score bonus so exact matches rank highest
            title = m.get("title", "").lower()
            if q in title:
                score += 1.0   # Strong boost for title substring match
            elif q in m.get("genre", "").lower():
                score += 0.5   # Moderate boost for genre substring match

            # 2. Semantic Plot Similarity:  cosine similarity between
            #    the query vector and the movie's pre-computed plot vector
            m_vec = embeddings.get(m.get("imdbID"))
            if m_vec is not None and query_norm > 0:
                m_norm = np.linalg.norm(m_vec)
                if m_norm > 0:
                    # Cosine similarity = dot(a,b) / (||a|| * ||b||)
                    cos_sim = np.dot(query_vec, m_vec) / (query_norm * m_norm)
                    score += max(0, cos_sim)  # Clamp negative similarities to 0

            # Only keep movies above a relevance threshold of 0.3
            if score > 0.3:
                scored_movies.append((score, m))

        # Sort by score descending — most relevant movies first
        scored_movies.sort(key=lambda x: x[0], reverse=True)
        matched = [m for score, m in scored_movies]

    else:
        # --- Fallback path: simple keyword matching ---
        # Used when embeddings or spaCy model aren't available
        for m in MOVIES_DB:
            title = m.get("title", "").lower()
            genre = m.get("genre", "").lower()
            plot = m.get("plot", "").lower()
            # Check if the query appears in title, genre, or plot text
            if q in title or q in genre or q in plot:
                matched.append(m)

    # --- Pagination: return 10 results per page ---
    start_idx = (page_num - 1) * 10   # 0-based start index
    end_idx = start_idx + 10           # Exclusive end index

    # Build the paginated response with minimal fields for the grid cards
    paginated_results = []
    for m in matched[start_idx:end_idx]:
        paginated_results.append({
            "imdbID": m.get("imdbID"),
            "title": m.get("title"),
            "year": m.get("year"),
            "poster": m.get("poster", ""),
        })

    # Return the page of results plus total count and search method used
    return jsonify({
        "movies": paginated_results,
        "totalResults": len(matched),
        "source": "semantic" if embeddings else "keyword_fallback"
    })


# ---------- Movie detail ----------
@app.route("/api/movie/<imdb_id>")
def get_movie_detail(imdb_id):
    """
    Get full movie details from the local dataset for a given IMDB ID.
    Performs a linear scan of MOVIES_DB to find the matching entry.

    URL param:
        imdb_id — the IMDB ID string (e.g., "tt0111161")

    Returns JSON with all movie fields or a 404 error.
    """
    for m in MOVIES_DB:
        if m.get("imdbID") == imdb_id:
            # Return a curated set of fields for the detail modal
            return jsonify({
                "imdbID": m.get("imdbID"),
                "title": m.get("title"),
                "year": m.get("year"),
                "genre": m.get("genre"),
                "plot": m.get("plot"),
                "poster": m.get("poster"),
                "rating": m.get("rating"),
                "director": m.get("director", ""),
                "actors": m.get("actors", ""),
                "runtime": m.get("runtime", ""),
                "rated": m.get("rated", ""),
            })

    # No movie matched the given IMDB ID
    return jsonify({"error": "Movie not found"}), 404


# ---------- Similar movies (semantic recommendations) ----------
@app.route("/api/similar/<imdb_id>")
def get_similar_movies(imdb_id):
    """
    Find the top 5 most similar movies based on cosine similarity
    of spaCy plot embeddings.

    Approach:
      1. Retrieve the target movie's pre-computed embedding vector
      2. Compute cosine similarity against every other movie's embedding
      3. Sort descending and return the top 5

    Returns JSON: { similar_movies: [ {imdbID, title, year, poster, similarity%}, ... ] }
    """
    # Ensure embeddings are loaded
    embeddings = MODELS.get("movie_embeddings")
    if not embeddings:
        return jsonify({"error": "Semantic embeddings not loaded"}), 500

    # Get the target movie's embedding vector
    target_vec = embeddings.get(imdb_id)
    if target_vec is None:
        return jsonify({"error": "Movie embedding not found"}), 404

    # Pre-compute the target vector's L2 norm for cosine similarity
    target_norm = np.linalg.norm(target_vec)
    if target_norm == 0:
        return jsonify({"error": "Invalid embedding"}), 500

    # Score every other movie by cosine similarity to the target
    scored_movies = []
    for m in MOVIES_DB:
        m_id = m.get("imdbID")
        if m_id == imdb_id:
            continue  # Skip the target movie itself

        m_vec = embeddings.get(m_id)
        if m_vec is not None:
            m_norm = np.linalg.norm(m_vec)
            if m_norm > 0:
                # Cosine similarity = dot(a,b) / (||a|| * ||b||)
                cos_sim = np.dot(target_vec, m_vec) / (target_norm * m_norm)
                scored_movies.append((cos_sim, m))

    # Sort by similarity score, highest first
    scored_movies.sort(key=lambda x: x[0], reverse=True)

    # Build the response list with the top 5 most similar movies
    similar_list = []
    for score, m in scored_movies[:5]:
        similar_list.append({
            "imdbID": m.get("imdbID"),
            "title": m.get("title"),
            "year": m.get("year"),
            "poster": m.get("poster", ""),
            "similarity": round(float(score) * 100, 1)  # Convert to percentage
        })

    return jsonify({"similar_movies": similar_list})


# ---------- EDA metrics ----------
@app.route("/api/eda")
def get_eda():
    """
    Return Exploratory Data Analysis metrics (genre distribution,
    top TF-IDF words per genre) from the pre-computed eda_metrics.json.
    """
    eda = MODELS.get("eda_metrics")
    if not eda:
        return jsonify({"error": "EDA metrics not loaded"}), 500
    return jsonify(eda)


# ---------- Genre classification (AI Analyze) ----------
@app.route("/api/analyze", methods=["POST"])
def analyze_plot():
    """
    Predict genres from a movie plot using all three trained classifiers
    (Naive Bayes, Logistic Regression, SVM) and return each model's
    predicted genre with its confidence percentage.

    Expects JSON body: { "plot": "..." }

    Pipeline:
      1. Preprocess the plot text (lowercase, strip punctuation)
      2. Transform through the fitted TF-IDF vectorizer → sparse vector
      3. Run each classifier's predict_proba() to get class probabilities
      4. Pick the class with highest probability as the prediction

    Returns JSON: { "Naive Bayes": {prediction, confidence}, ... }
    """
    # Parse the JSON request body
    data = request.json
    plot = data.get("plot", "")

    if not plot:
        return jsonify({"error": "No plot provided"}), 400

    # Retrieve the required model components
    vec, le = MODELS.get("vectorizer"), MODELS.get("label_encoder")
    nb, lr, svm = MODELS.get("nb"), MODELS.get("lr"), MODELS.get("svm")

    # Verify all models are loaded
    if not all([vec, le, nb, lr, svm]):
        return jsonify({"error": "Models not loaded"}), 500

    try:
        # Step 1: Clean the input plot text
        clean_plot = preprocess(plot)

        # Step 2: Transform to TF-IDF feature vector (same vocabulary as training)
        X_vec = vec.transform([clean_plot])

        results = {}

        # --- Naive Bayes prediction ---
        nb_prob = nb.predict_proba(X_vec)[0]        # Get probability for each class
        nb_pred_idx = np.argmax(nb_prob)             # Index of highest-probability class
        results["Naive Bayes"] = {
            "prediction": str(le.inverse_transform([nb_pred_idx])[0]),  # Map index → genre name
            "confidence": round(float(nb_prob[nb_pred_idx]) * 100, 1)   # Convert to percentage
        }

        # --- Logistic Regression prediction ---
        lr_prob = lr.predict_proba(X_vec)[0]
        lr_pred_idx = np.argmax(lr_prob)
        results["Logistic Regression"] = {
            "prediction": str(le.inverse_transform([lr_pred_idx])[0]),
            "confidence": round(float(lr_prob[lr_pred_idx]) * 100, 1)
        }

        # --- SVM prediction (via CalibratedClassifierCV for probabilities) ---
        svm_prob = svm.predict_proba(X_vec)[0]
        svm_pred_idx = np.argmax(svm_prob)
        results["SVM"] = {
            "prediction": str(le.inverse_transform([svm_pred_idx])[0]),
            "confidence": round(float(svm_prob[svm_pred_idx]) * 100, 1)
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------- Word relevance ----------
@app.route("/api/word-relevance", methods=["POST"])
def word_relevance():
    """
    Measure how relevant a keyword is to a movie's plot using spaCy
    semantic vector similarity.

    Expects JSON body: { "word": "...", "plot": "..." }

    Approach:
      1. Convert both the word and the full plot into spaCy Doc objects
      2. Check if the word has a vector (is in spaCy's vocabulary)
      3. Compute .similarity() which uses cosine similarity on averaged
         word vectors

    Returns JSON: { word, score (0-100%), models: {...} }
    """
    # Parse inputs from JSON body
    data = request.json
    word = data.get("word", "").strip()
    plot = data.get("plot", "").strip()

    # Validate that both fields are provided
    if not word or not plot:
        return jsonify({"error": "Word or plot missing"}), 400

    # Ensure spaCy model is available
    if not NLP:
        return jsonify({"error": "spaCy model not loaded"}), 500

    try:
        # Process word and plot through the spaCy pipeline to get Doc objects
        word_doc = NLP(word)
        plot_doc = NLP(plot)

        # Check if the word is Out-Of-Vocabulary (OOV)
        # OOV words have no vector and can't be meaningfully compared
        if not word_doc.has_vector:
            return jsonify({
                "word": word,
                "score": 0,
                "models": {"Word in Vocab": "No (0%)"},
                "message": f"'{word}' is not in the spaCy vocabulary."
            })

        # Compute semantic similarity between word and entire plot
        # spaCy's .similarity() uses cosine similarity on averaged word vectors
        similarity = word_doc.similarity(plot_doc)

        # Scale the similarity (typically -1 to 1) to a 0-100% range
        score_pct = round(max(0, similarity) * 100, 1)

        return jsonify({
            "word": word,
            "score": score_pct,
            "models": {
                "spaCy EnCoreWebMd Vector Match": f"{score_pct}% Match"
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------- Model evaluation metrics ----------
@app.route("/api/metrics")
def get_metrics():
    """
    Return the pre-computed model evaluation metrics (accuracy, precision,
    recall, F1, confusion matrices, hyperparameters) from metrics.json.
    Used by the dashboard to render performance comparison charts.
    """
    metrics = MODELS.get("metrics")
    if not metrics:
        return jsonify({"error": "Metrics not found"}), 404
    return jsonify({"metrics": metrics})


# ============================================================
# STEP 5: Application entry point
# ============================================================
if __name__ == "__main__":
    # Start the Flask development server with hot-reload enabled
    # debug=True enables auto-restart on code changes + detailed error pages
    app.run(debug=True)
