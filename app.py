import os
import re
import json
import pickle
import numpy as np
import spacy
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- Load Dataset at startup ---
MOVIES_DB = []
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

import csv

try:
    print("Loading movies dataset from CSV...")
    data_path = os.path.join(DATA_DIR, "movies.csv")
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "genres_list" in row and isinstance(row["genres_list"], str):
                row["genres_list"] = row["genres_list"].split("|")
            MOVIES_DB.append(row)
    print(f"✅ Loaded {len(MOVIES_DB)} movies from CSV.")
except Exception as e:
    print(f"⚠️ Error loading CSV: {e}")

# --- Load models at startup ---
def load_models():
    models = {}
    try:
        with open(os.path.join(MODELS_DIR, "nb_model.pkl"), "rb") as f:
            models["nb"] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "lr_model.pkl"), "rb") as f:
            models["lr"] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "svm_model.pkl"), "rb") as f:
            models["svm"] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "vectorizer.pkl"), "rb") as f:
            models["vectorizer"] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "rb") as f:
            models["label_encoder"] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "metrics.json"), "r") as f:
            models["metrics"] = json.load(f)
        with open(os.path.join(MODELS_DIR, "genre_classes.json"), "r") as f:
            models["genre_classes"] = json.load(f)
        with open(os.path.join(MODELS_DIR, "movie_embeddings.pkl"), "rb") as f:
            models["movie_embeddings"] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "eda_metrics.json"), "r") as f:
            models["eda_metrics"] = json.load(f)
        print("✅ All models, embeddings, and EDA metrics loaded successfully!")
    except FileNotFoundError as e:
        print(f"⚠️  Model files not found: {e}")
        print("   Run fetch_data.py then train_models.py first.")
    return models

MODELS = load_models()

try:
    print("Loading spaCy model...")
    NLP = spacy.load("en_core_web_md")
    print("✅ spaCy model loaded!")
except OSError:
    print("⚠️  spaCy model en_core_web_md not found.")
    NLP = None


def preprocess(text):
    """Clean text for TF-IDF."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


# ========== ROUTES ==========

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search")
def search_movies():
    """Search movies using Semantic Search (Cosine Similarity) with fallback to local exact match."""
    query = request.args.get("q", "").strip()
    page = request.args.get("page", "1")
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
        
    try:
        page_num = int(page)
    except ValueError:
        page_num = 1
        
    q = query.lower()
    matched = []
    
    # Calculate Semantic Similarity if embeddings are loaded
    embeddings = MODELS.get("movie_embeddings")
    if embeddings and NLP:
        query_vec = NLP(query).vector
        query_norm = np.linalg.norm(query_vec)
        
        scored_movies = []
        for m in MOVIES_DB:
            score = 0
            # 1. Exact match boost for Title/Genre
            title = m.get("title", "").lower()
            if q in title:
                score += 1.0  # Big boost for title exact match
            elif q in m.get("genre", "").lower():
                score += 0.5  # Boost for genre exact match
                
            # 2. Semantic Plot Similarity
            m_vec = embeddings.get(m.get("imdbID"))
            if m_vec is not None and query_norm > 0:
                m_norm = np.linalg.norm(m_vec)
                if m_norm > 0:
                    cos_sim = np.dot(query_vec, m_vec) / (query_norm * m_norm)
                    score += max(0, cos_sim) # Range 0 to 1
                    
            if score > 0.3: # Threshold
                scored_movies.append((score, m))
                
        # Sort descending by score
        scored_movies.sort(key=lambda x: x[0], reverse=True)
        matched = [m for score, m in scored_movies]
    else:
        # Fallback keyword match
        for m in MOVIES_DB:
            title = m.get("title", "").lower()
            genre = m.get("genre", "").lower()
            plot = m.get("plot", "").lower()
            if q in title or q in genre or q in plot:
                matched.append(m)
                
    # Pagination (10 per page)
    start_idx = (page_num - 1) * 10
    end_idx = start_idx + 10
    paginated_results = []
    for m in matched[start_idx:end_idx]:
        paginated_results.append({
            "imdbID": m.get("imdbID"),
            "title": m.get("title"),
            "year": m.get("year"),
            "poster": m.get("poster", ""),
        })
    
    return jsonify({
        "movies": paginated_results,
        "totalResults": len(matched),
        "source": "semantic" if embeddings else "keyword_fallback"
    })

@app.route("/api/movie/<imdb_id>")
def get_movie_detail(imdb_id):
    """Get full movie details from local dataset."""
    for m in MOVIES_DB:
        if m.get("imdbID") == imdb_id:
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
    return jsonify({"error": "Movie not found"}), 404

@app.route("/api/similar/<imdb_id>")
def get_similar_movies(imdb_id):
    """Find top 5 similar movies based on semantic plot embeddings."""
    embeddings = MODELS.get("movie_embeddings")
    if not embeddings:
        return jsonify({"error": "Semantic embeddings not loaded"}), 500
        
    target_vec = embeddings.get(imdb_id)
    if target_vec is None:
        return jsonify({"error": "Movie embedding not found"}), 404
        
    target_norm = np.linalg.norm(target_vec)
    if target_norm == 0:
        return jsonify({"error": "Invalid embedding"}), 500
        
    scored_movies = []
    for m in MOVIES_DB:
        m_id = m.get("imdbID")
        if m_id == imdb_id: continue # Skip itself
        
        m_vec = embeddings.get(m_id)
        if m_vec is not None:
            m_norm = np.linalg.norm(m_vec)
            if m_norm > 0:
                cos_sim = np.dot(target_vec, m_vec) / (target_norm * m_norm)
                scored_movies.append((cos_sim, m))
                
    # Sort descending
    scored_movies.sort(key=lambda x: x[0], reverse=True)
    
    similar_list = []
    for score, m in scored_movies[:5]: # Top 5
        similar_list.append({
            "imdbID": m.get("imdbID"),
            "title": m.get("title"),
            "year": m.get("year"),
            "poster": m.get("poster", ""),
            "similarity": round(float(score) * 100, 1)
        })
        
    return jsonify({"similar_movies": similar_list})

@app.route("/api/eda")
def get_eda():
    """Return EDA metrics from the models folder."""
    eda = MODELS.get("eda_metrics")
    if not eda:
        return jsonify({"error": "EDA metrics not loaded"}), 500
    return jsonify(eda)


@app.route("/api/analyze", methods=["POST"])
def analyze_plot():
    """Predict genres and return confidence using loaded models."""
    data = request.json
    plot = data.get("plot", "")
    
    if not plot:
        return jsonify({"error": "No plot provided"}), 400
        
    vec, le = MODELS.get("vectorizer"), MODELS.get("label_encoder")
    nb, lr, svm = MODELS.get("nb"), MODELS.get("lr"), MODELS.get("svm")
    
    if not all([vec, le, nb, lr, svm]):
        return jsonify({"error": "Models not loaded"}), 500
        
    try:
        clean_plot = preprocess(plot)
        X_vec = vec.transform([clean_plot])
        
        results = {}
        # Naive Bayes
        nb_prob = nb.predict_proba(X_vec)[0]
        nb_pred_idx = np.argmax(nb_prob)
        results["Naive Bayes"] = {
            "prediction": str(le.inverse_transform([nb_pred_idx])[0]),
            "confidence": round(float(nb_prob[nb_pred_idx]) * 100, 1)
        }
        
        # Logistic Regression
        lr_prob = lr.predict_proba(X_vec)[0]
        lr_pred_idx = np.argmax(lr_prob)
        results["Logistic Regression"] = {
            "prediction": str(le.inverse_transform([lr_pred_idx])[0]),
            "confidence": round(float(lr_prob[lr_pred_idx]) * 100, 1)
        }
        
        # SVM (Calibrated)
        svm_prob = svm.predict_proba(X_vec)[0]
        svm_pred_idx = np.argmax(svm_prob)
        results["SVM"] = {
            "prediction": str(le.inverse_transform([svm_pred_idx])[0]),
            "confidence": round(float(svm_prob[svm_pred_idx]) * 100, 1)
        }
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
        
@app.route("/api/word-relevance", methods=["POST"])
def word_relevance():
    """Finds how relevant a keyword is to the movie plot using spaCy semantic similarity."""
    data = request.json
    word = data.get("word", "").strip()
    plot = data.get("plot", "").strip()
    
    if not word or not plot:
        return jsonify({"error": "Word or plot missing"}), 400
        
    if not NLP:
        return jsonify({"error": "spaCy model not loaded"}), 500
        
    try:
        word_doc = NLP(word)
        plot_doc = NLP(plot)
        
        # Check if the word is Out-Of-Vocabulary
        if not word_doc.has_vector:
            return jsonify({
                "word": word,
                "score": 0,
                "models": {"Word in Vocab": "No (0%)"},
                "message": f"'{word}' is not in the spaCy vocabulary."
            })
            
        similarity = word_doc.similarity(plot_doc)
        
        # Scale to a percentage for UI
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


@app.route("/api/metrics")
def get_metrics():
    """Return model evaluation metrics."""
    metrics = MODELS.get("metrics")
    if not metrics:
        return jsonify({"error": "Metrics not found"}), 404
    return jsonify({"metrics": metrics})


if __name__ == "__main__":
    app.run(debug=True)
