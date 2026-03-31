import os
import json
import csv
import pickle
import numpy as np
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV

def load_data():
    data_path = os.path.join(os.path.dirname(__file__), "data", "movies.csv")
    movies = []
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "genres_list" in row and isinstance(row["genres_list"], str):
                row["genres_list"] = row["genres_list"].split("|")
            movies.append(row)
    return movies

def preprocess(text):
    """Clean plot text for TF-IDF."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def main():
    print("Loading movie data...")
    movies = load_data()
    print(f"Loaded {len(movies)} movies")
    
    # Extract plots and primary genres
    raw_plots = []
    primary_genres = []
    
    for m in movies:
        plot = m.get("plot", "")
        gl = m.get("genres_list", [])
        
        if plot and gl and len(gl) > 0:
            # Consolidate rare/ambiguous genres for better class balance
            genre = gl[0].strip()
            
            # Grouping logical categories
            if genre in ["Sci-Fi", "Mystery", "Thriller"]:
                genre = "Drama" # Or potentially Action depending on exact counts
            elif genre in ["Family", "Music", "Musical"]:
                genre = "Comedy"
            elif genre in ["Documentary", "Short", "History", "Biography", "Sport", "War", "Western"]:
                # If the first genre is too rare, use the second genre if it exists
                if len(gl) > 1:
                    genre = gl[1].strip()
                else:
                    genre = "Drama" # Fallback wrapper
                    
            raw_plots.append(plot)
            primary_genres.append(genre)
            
    # Remove extremely sparse classes to allow StratifiedKFold cross-validation (requires >= n_splits samples)
    genre_counts = Counter(primary_genres)
    valid_genres = {g for g, count in genre_counts.items() if count >= 10}
    
    plots = []
    genres = []
    for p, g in zip(raw_plots, primary_genres):
        if g in valid_genres:
            plots.append(p)
            genres.append(g)
            
    print(f"Usable samples (after filtering rare genres): {len(plots)}")
    print(f"Genre distribution: {Counter(genres).most_common()}")

    # --- 1. Label Encoding ---
    le = LabelEncoder()
    y = le.fit_transform(genres)
    print(f"Genre classes: {le.classes_}")
    
    # --- 2. TF-IDF Vectorization ---
    print("Cleaning and Vectorizing plots...")
    cleaned_plots = [preprocess(p) for p in plots]
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(cleaned_plots)
    
    # --- 3. Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}\n")
    
    # --- 4. Model Training & Tuning ---
    
    # Model 1: Naive Bayes
    print("[1/3] Training Naive Bayes...")
    nb_grid = GridSearchCV(
        MultinomialNB(),
        {'alpha': [0.1, 0.5, 1.0, 2.0]},
        cv=5, scoring='accuracy', n_jobs=-1
    )
    nb_grid.fit(X_train, y_train)
    nb = nb_grid.best_estimator_
    y_pred_nb = nb.predict(X_test)
    print(f"  Best alpha: {nb_grid.best_params_['alpha']}")
    print(f"  CV Accuracy: {nb_grid.best_score_:.4f}")
    print(f"  Test Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}\n")
    
    # Model 2: Logistic Regression (balanced class weight)
    print("[2/3] Training Logistic Regression...")
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, class_weight='balanced'),
        {'C': [0.1, 1.0, 5.0, 10.0]},
        cv=5, scoring='accuracy', n_jobs=-1
    )
    lr_grid.fit(X_train, y_train)
    lr = lr_grid.best_estimator_
    y_pred_lr = lr.predict(X_test)
    print(f"  Best C: {lr_grid.best_params_['C']}")
    print(f"  CV Accuracy: {lr_grid.best_score_:.4f}")
    print(f"  Test Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}\n")
    
    # Model 3: Linear SVM (balanced class weight)
    print("[3/3] Training SVM...")
    svm_grid = GridSearchCV(
        LinearSVC(max_iter=2000, class_weight='balanced', dual="auto"),
        {'C': [0.01, 0.1, 1.0, 5.0, 10.0]},
        cv=5, scoring='accuracy', n_jobs=-1
    )
    svm_grid.fit(X_train, y_train)
    # Wrap in CalibratedClassifierCV for probability outputs
    svm_cal = CalibratedClassifierCV(svm_grid.best_estimator_, cv=3)
    svm_cal.fit(X_train, y_train)
    y_pred_svm = svm_cal.predict(X_test)
    print(f"  Best C: {svm_grid.best_params_['C']}")
    print(f"  CV Accuracy: {svm_grid.best_score_:.4f}")
    print(f"  Test Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
    
    # --- Compute comprehensive metrics ---
    metrics = {}
    for name, y_pred, grid in [
        ("Naive Bayes", y_pred_nb, nb_grid),
        ("Logistic Regression", y_pred_lr, lr_grid),
        ("SVM", y_pred_svm, svm_grid)
    ]:
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            "cv_accuracy": float(grid.best_score_),
            "best_params": grid.best_params_,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        
    # --- Generate Semantic Embeddings & EDA Metrics ---
    import spacy
    print("\n[4/4] Generating Embeddings & EDA Metrics...")
    print("  Loading spaCy (en_core_web_md)...")
    nlp = spacy.load("en_core_web_md")
    
    embeddings = {}
    all_genres = []
    for m in movies:
        plot = m.get("plot", "")
        embeddings[m["imdbID"]] = nlp(plot).vector
        gl = m.get("genres_list", [])
        if isinstance(gl, list):
            all_genres.extend(gl)
            
    eda_metrics = {
        "genre_distribution": dict(Counter(all_genres).most_common()),
        "top_words": {}
    }
    
    feature_names = vectorizer.get_feature_names_out()
    for g_idx, g_name in enumerate(le.classes_):
        class_rows = X[y == g_idx]
        if class_rows.shape[0] > 0:
            avg_tfidf = np.asarray(class_rows.mean(axis=0)).flatten()
            # Sort and get top indices
            # argsort returns lowest to highest, so we take the last 15 and reverse
            top_indices = avg_tfidf.argsort()[-15:][::-1]
            eda_metrics["top_words"][str(g_name)] = [{"text": feature_names[i], "value": float(avg_tfidf[i])} for i in top_indices]
            
    # --- Save everything ---
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    with open(os.path.join(models_dir, "nb_model.pkl"), "wb") as f:
        pickle.dump(nb, f)
    with open(os.path.join(models_dir, "lr_model.pkl"), "wb") as f:
        pickle.dump(lr, f)
    with open(os.path.join(models_dir, "svm_model.pkl"), "wb") as f:
        pickle.dump(svm_cal, f)
    with open(os.path.join(models_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(models_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(models_dir, "genre_classes.json"), "w") as f:
        json.dump(list(le.classes_), f)
    with open(os.path.join(models_dir, "movie_embeddings.pkl"), "wb") as f:
        pickle.dump(embeddings, f)
    with open(os.path.join(models_dir, "eda_metrics.json"), "w") as f:
        json.dump(eda_metrics, f, indent=2)
    
    print("\n✅ All models saved to models/ directory!")
    print("   - nb_model.pkl, lr_model.pkl, svm_model.pkl")
    print("   - vectorizer.pkl, label_encoder.pkl")
    print("   - metrics.json, genre_classes.json")
    print("   - movie_embeddings.pkl, eda_metrics.json")

if __name__ == "__main__":
    main()
