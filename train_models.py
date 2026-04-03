# ============================================================
# train_models.py — Model Training Pipeline for CineNLP
#
# PURPOSE:
#   Trains three genre classification models (Naive Bayes,
#   Logistic Regression, SVM) on movie plot text, generates
#   spaCy semantic embeddings for every movie, computes EDA
#   metrics, and saves everything to the models/ directory.
#
# USAGE:
#   Run AFTER fetch_data.py has populated data/movies.csv:
#       python train_models.py
#
# OUTPUT FILES (in models/):
#   nb_model.pkl         — trained Naive Bayes classifier
#   lr_model.pkl         — trained Logistic Regression classifier
#   svm_model.pkl        — calibrated SVM classifier
#   vectorizer.pkl       — fitted TF-IDF vectorizer
#   label_encoder.pkl    — fitted LabelEncoder (genre ↔ int)
#   metrics.json         — accuracy/precision/recall/F1/confusion matrices
#   genre_classes.json   — ordered list of genre class names
#   movie_embeddings.pkl — spaCy 300-dim plot vectors per movie
#   eda_metrics.json     — genre distribution + top TF-IDF words
# ============================================================

# --- Standard library imports ---
import os       # File path operations
import json     # Read/write JSON files
import csv      # Parse the CSV dataset
import pickle   # Serialise trained models to disk
import re       # Regex for text cleaning

# --- Third-party imports ---
import numpy as np           # Numerical operations (array math, argmax)
from collections import Counter  # Count genre frequencies

# --- Scikit-learn imports ---
from sklearn.feature_extraction.text import TfidfVectorizer   # Convert text → TF-IDF sparse matrix
from sklearn.preprocessing import LabelEncoder                # Map genre strings → integer labels
from sklearn.model_selection import train_test_split, GridSearchCV  # Split data & hyperparameter tuning
from sklearn.naive_bayes import MultinomialNB                 # Naive Bayes classifier
from sklearn.linear_model import LogisticRegression            # Logistic Regression classifier
from sklearn.svm import LinearSVC                             # Linear Support Vector Classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# ↑ Evaluation metrics for measuring classifier performance
from sklearn.calibration import CalibratedClassifierCV
# ↑ Wraps LinearSVC to enable predict_proba() (SVM doesn't natively support probabilities)


# ============================================================
# STEP 1: Load the movie dataset from CSV
# ============================================================

def load_data():
    """
    Read data/movies.csv and return a list of dicts (one per movie).
    Converts the pipe-delimited genres_list column back into a Python list.
    """
    data_path = os.path.join(os.path.dirname(__file__), "data", "movies.csv")
    movies = []

    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)  # Each row becomes a dict keyed by CSV headers
        for row in reader:
            # Restore the genres_list from pipe-delimited string to a list
            # e.g., "Action|Drama|Thriller" → ["Action", "Drama", "Thriller"]
            if "genres_list" in row and isinstance(row["genres_list"], str):
                row["genres_list"] = row["genres_list"].split("|")
            movies.append(row)

    return movies


# ============================================================
# STEP 2: Text preprocessing
# ============================================================

def preprocess(text):
    """
    Clean plot text for TF-IDF vectorization.
    Steps:
      1. Lowercase — normalises case ("Action" vs "action")
      2. Remove non-alphabetic chars — strips numbers, punctuation, symbols
    """
    text = text.lower()                        # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)    # Keep only letters and whitespace
    return text


# ============================================================
# STEP 3: Main training pipeline
# ============================================================

def main():
    # ------- 3a: Load and inspect data -------
    print("Loading movie data...")
    movies = load_data()
    print(f"Loaded {len(movies)} movies")

    # ------- 3b: Extract plots and assign primary genres -------
    # Each movie may have multiple genres; we use the FIRST genre as
    # the primary label for classification (single-label task).
    raw_plots = []       # Will hold the raw plot text strings
    primary_genres = []  # Will hold the corresponding genre label for each plot

    for m in movies:
        plot = m.get("plot", "")
        gl = m.get("genres_list", [])

        # Only use movies that have both a plot and at least one genre
        if plot and gl and len(gl) > 0:
            genre = gl[0].strip()  # Take the first (primary) genre

            # --- Genre consolidation ---
            # Rare or ambiguous genres are merged into larger umbrella categories
            # to improve class balance and model stability.

            # Sci-Fi, Mystery, Thriller → merged into "Drama" (thematic overlap)
            if genre in ["Sci-Fi", "Mystery", "Thriller"]:
                genre = "Drama"

            # Family, Music, Musical → merged into "Comedy" (tonal overlap)
            elif genre in ["Family", "Music", "Musical"]:
                genre = "Comedy"

            # Very rare genres → try using the second genre if available,
            # otherwise fall back to "Drama"
            elif genre in ["Documentary", "Short", "History", "Biography", "Sport", "War", "Western"]:
                if len(gl) > 1:
                    genre = gl[1].strip()  # Use the second genre instead
                else:
                    genre = "Drama"        # Fallback for single-genre rare movies

            raw_plots.append(plot)
            primary_genres.append(genre)

    # ------- 3c: Filter out sparse genres -------
    # StratifiedKFold cross-validation requires at least n_splits (5) samples
    # per class.  We require at least 10 samples per genre for stability.
    genre_counts = Counter(primary_genres)
    valid_genres = {g for g, count in genre_counts.items() if count >= 10}

    # Keep only samples whose genre passed the frequency filter
    plots = []
    genres = []
    for p, g in zip(raw_plots, primary_genres):
        if g in valid_genres:
            plots.append(p)
            genres.append(g)

    print(f"Usable samples (after filtering rare genres): {len(plots)}")
    print(f"Genre distribution: {Counter(genres).most_common()}")

    # ============================================================
    # STEP 4: Feature engineering
    # ============================================================

    # --- 4a: Label Encoding ---
    # Convert genre strings ("Drama", "Comedy", ...) → integers (0, 1, ...)
    # Needed because sklearn classifiers work with numeric labels.
    le = LabelEncoder()
    y = le.fit_transform(genres)  # Fit on training genres and transform
    print(f"Genre classes: {le.classes_}")

    # --- 4b: TF-IDF Vectorization ---
    # Transform raw plot text into a numerical sparse matrix where each
    # column represents a word (or bigram) and each value is its TF-IDF score.
    print("Cleaning and Vectorizing plots...")
    cleaned_plots = [preprocess(p) for p in plots]  # Apply text cleaning to all plots

    vectorizer = TfidfVectorizer(
        max_features=5000,        # Keep top 5000 most important terms
        stop_words='english',     # Remove common English stop words ("the", "is", etc.)
        ngram_range=(1, 2)        # Include both unigrams ("action") and bigrams ("car chase")
    )
    X = vectorizer.fit_transform(cleaned_plots)  # Fit vocabulary & transform to sparse matrix

    # ============================================================
    # STEP 5: Train/Test Split
    # ============================================================
    # Split data 80/20 for training/testing with stratification
    # (preserves genre proportions in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,      # 20% held out for testing
        random_state=42,     # Fixed seed for reproducibility
        stratify=y           # Maintain class distribution in both splits
    )
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}\n")

    # ============================================================
    # STEP 6: Model Training with Hyperparameter Tuning
    # ============================================================
    # Each model is tuned via GridSearchCV (exhaustive grid search with
    # 5-fold stratified cross-validation on the training set).

    # --- Model 1: Multinomial Naive Bayes ---
    # Good baseline for text classification; alpha controls Laplace smoothing
    print("[1/3] Training Naive Bayes...")
    nb_grid = GridSearchCV(
        MultinomialNB(),
        {'alpha': [0.1, 0.5, 1.0, 2.0]},  # Smoothing parameter grid
        cv=5,                               # 5-fold cross-validation
        scoring='accuracy',                 # Optimise for accuracy
        n_jobs=-1                           # Use all CPU cores
    )
    nb_grid.fit(X_train, y_train)           # Run the grid search
    nb = nb_grid.best_estimator_            # Extract the best model
    y_pred_nb = nb.predict(X_test)          # Predictions on test set
    print(f"  Best alpha: {nb_grid.best_params_['alpha']}")
    print(f"  CV Accuracy: {nb_grid.best_score_:.4f}")
    print(f"  Test Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}\n")

    # --- Model 2: Logistic Regression ---
    # Linear model with L2 regularisation; C controls regularisation strength
    # class_weight='balanced' adjusts weights inversely proportional to class frequencies
    print("[2/3] Training Logistic Regression...")
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        {'C': [0.1, 1.0, 5.0, 10.0]},      # Inverse regularisation strength grid
        cv=5, scoring='accuracy', n_jobs=-1
    )
    lr_grid.fit(X_train, y_train)
    lr = lr_grid.best_estimator_
    y_pred_lr = lr.predict(X_test)
    print(f"  Best C: {lr_grid.best_params_['C']}")
    print(f"  CV Accuracy: {lr_grid.best_score_:.4f}")
    print(f"  Test Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}\n")

    # --- Model 3: Linear SVM (Support Vector Machine) ---
    # LinearSVC is faster than SVC for high-dimensional sparse data like TF-IDF
    # Wrapped in CalibratedClassifierCV to enable predict_proba()
    print("[3/3] Training SVM...")
    svm_grid = GridSearchCV(
        LinearSVC(max_iter=2000, class_weight='balanced', dual="auto"),
        {'C': [0.01, 0.1, 1.0, 5.0, 10.0]},  # Regularisation parameter grid
        cv=5, scoring='accuracy', n_jobs=-1
    )
    svm_grid.fit(X_train, y_train)

    # Wrap the best SVM in CalibratedClassifierCV to get probability estimates
    # (LinearSVC only outputs decision values, not probabilities)
    svm_cal = CalibratedClassifierCV(svm_grid.best_estimator_, cv=3)
    svm_cal.fit(X_train, y_train)       # Re-fit the calibrated wrapper
    y_pred_svm = svm_cal.predict(X_test)
    print(f"  Best C: {svm_grid.best_params_['C']}")
    print(f"  CV Accuracy: {svm_grid.best_score_:.4f}")
    print(f"  Test Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")

    # ============================================================
    # STEP 7: Compute comprehensive evaluation metrics
    # ============================================================
    # For each model, calculate accuracy, weighted precision/recall/F1,
    # cross-validation accuracy, best hyperparameters, and confusion matrix.
    metrics = {}
    for name, y_pred, grid in [
        ("Naive Bayes", y_pred_nb, nb_grid),
        ("Logistic Regression", y_pred_lr, lr_grid),
        ("SVM", y_pred_svm, svm_grid)
    ]:
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            # Weighted average accounts for class imbalance
            "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            "cv_accuracy": float(grid.best_score_),      # Best cross-validation score
            "best_params": grid.best_params_,             # Winning hyperparameters
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),  # Serialisable matrix
        }

    # ============================================================
    # STEP 8: Generate semantic embeddings & EDA metrics
    # ============================================================
    import spacy
    print("\n[4/4] Generating Embeddings & EDA Metrics...")
    print("  Loading spaCy (en_core_web_md)...")

    # Load the medium English model which includes 300-dim word vectors
    nlp = spacy.load("en_core_web_md")

    # --- 8a: Compute plot embeddings for every movie ---
    # Each movie's plot is converted to a 300-dim vector (average of word vectors)
    # These embeddings power semantic search and similar-movie recommendations
    embeddings = {}
    all_genres = []  # Collect all genre occurrences for distribution stats
    for m in movies:
        plot = m.get("plot", "")
        embeddings[m["imdbID"]] = nlp(plot).vector  # 300-dim numpy array (avg of word vectors)

        # Collect genre labels for EDA distribution chart
        gl = m.get("genres_list", [])
        if isinstance(gl, list):
            all_genres.extend(gl)

    # --- 8b: Compute EDA (Exploratory Data Analysis) metrics ---
    eda_metrics = {
        # Genre distribution: how many movies per genre across the dataset
        "genre_distribution": dict(Counter(all_genres).most_common()),
        # Top TF-IDF words: most distinctive terms for each genre class
        "top_words": {}
    }

    # For each genre class, find the top 15 words by average TF-IDF score
    feature_names = vectorizer.get_feature_names_out()  # Get the vocabulary terms
    for g_idx, g_name in enumerate(le.classes_):
        # Select only the TF-IDF rows belonging to this genre class
        class_rows = X[y == g_idx]
        if class_rows.shape[0] > 0:
            # Compute the column-wise mean TF-IDF score across all rows in this class
            avg_tfidf = np.asarray(class_rows.mean(axis=0)).flatten()

            # Get indices of the top 15 features by average TF-IDF value
            # argsort() returns ascending order, so we take the last 15 and reverse
            top_indices = avg_tfidf.argsort()[-15:][::-1]

            # Build a list of {text, value} dicts for the frontend word cloud
            eda_metrics["top_words"][str(g_name)] = [
                {"text": feature_names[i], "value": float(avg_tfidf[i])}
                for i in top_indices
            ]

    # ============================================================
    # STEP 9: Save all trained models and artifacts to disk
    # ============================================================
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)  # Create the models/ directory if it doesn't exist

    # Pickle the trained classifiers
    with open(os.path.join(models_dir, "nb_model.pkl"), "wb") as f:
        pickle.dump(nb, f)         # Naive Bayes
    with open(os.path.join(models_dir, "lr_model.pkl"), "wb") as f:
        pickle.dump(lr, f)         # Logistic Regression
    with open(os.path.join(models_dir, "svm_model.pkl"), "wb") as f:
        pickle.dump(svm_cal, f)    # Calibrated SVM (with predict_proba support)

    # Pickle the feature engineering artifacts
    with open(os.path.join(models_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)  # TF-IDF vectorizer (vocabulary + IDF weights)
    with open(os.path.join(models_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)          # Label encoder (genre ↔ integer mapping)

    # Save evaluation metrics as human-readable JSON
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save the ordered genre class names
    with open(os.path.join(models_dir, "genre_classes.json"), "w") as f:
        json.dump(list(le.classes_), f)

    # Pickle the semantic embeddings (dict: imdbID → numpy vector)
    with open(os.path.join(models_dir, "movie_embeddings.pkl"), "wb") as f:
        pickle.dump(embeddings, f)

    # Save EDA metrics as JSON for the dashboard visualizations
    with open(os.path.join(models_dir, "eda_metrics.json"), "w") as f:
        json.dump(eda_metrics, f, indent=2)

    # --- Print summary of saved files ---
    print("\n✅ All models saved to models/ directory!")
    print("   - nb_model.pkl, lr_model.pkl, svm_model.pkl")
    print("   - vectorizer.pkl, label_encoder.pkl")
    print("   - metrics.json, genre_classes.json")
    print("   - movie_embeddings.pkl, eda_metrics.json")


# Standard Python entry point guard
if __name__ == "__main__":
    main()
