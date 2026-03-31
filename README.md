# CineNLP — AI Movie Intelligence Dashboard

![CineNLP](https://img.shields.io/badge/Status-Complete-success.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web%2  Framework-black.svg)
![Machine Learning](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)
![NLP](https://img.shields.io/badge/spaCy-NLP-blueviolet.svg)

CineNLP is a comprehensive, end-to-end Machine Learning web application designed to analyze, classify, and intelligently search through a database of movies. Moving entirely beyond simple keyword matching, CineNLP employs advanced **Semantic Search** and **TF-IDF Vectorization** to recommend similar movies, predict movie genres solely from reading the plot summary, and analyze the contextual relevance of words to specific films.

---

## 🚀 Features

### 1. Semantic Movie Search
Instead of strict keyword matching, the search bar transforms your query into a mathematical vector and compares it against pre-computed 300-dimensional **spaCy embeddings** (`en_core_web_md`) of every movie plot in the database.
- Easily search abstract concepts like *"dream heist"* or *"space war"* and get highly accurate semantic matches using Cosine Similarity.

### 2. AI Genre Classification (NLP)
When viewing a movie, 3 separate Machine Learning models "read" the movie's plot and try to predict its primary genre using **TF-IDF (Term Frequency - Inverse Document Frequency)**. The dashboard displays the real-time confidence scores (probabilities) from each model.

### 3. "Similar Movies" Recommendation Engine
Every movie detail card includes a scrollable carousel of the top 5 most semantically similar movies. This is calculated live by matching the Cosine Similarity of the target movie's plot vector against all other plots in the dataset.

### 4. Interactive EDA & Performance Dashboard
A dedicated "Models" tab visualizes the underlying dataset and model performance metrics using `Chart.js`:
- **Model Evaluation:** Bar charts comparing CV Accuracy, Test Accuracy, Precision, Recall, and F1 Scores across the three models.
- **Dataset Distribution:** A donut chart showing the balance of the `movies.csv` genre classes.
- **Top Feature Words (TF-IDF):** An interactive word cloud that shows the exact vocabulary the machine learning models use to classify an *"Action"* movie vs a *"Comedy"* movie.

### 5. Word Context Relevance Checker
Allows the user to type an arbitrary word and checks how semantically relevant that word is to the currently viewed movie's plot using spaCy vector similarity computation.

---

## 🧠 Machine Learning Architecture

### The Dataset
The project is built on an offline dataset (`/data/movies.csv`) containing **847 distinct movies** scraped directly from the OMDB API.

### Preprocessing & NLP Pipeline
- **Text Cleaning:** All plots are lowercased and stripped of non-alphabetic characters using Regex.
- **Vectorization (TF-IDF):** Plots are transformed into a matrix of token counts containing the top 5,000 unigram and bigram features (`max_features=5000, ngram_range=(1,2)`).
- **Embeddings:** `spaCy` generates high-dimensional vectors for semantic comparisons.

### The Models
We use `GridSearchCV` with 5-fold cross-validation (`cv=5`) to optimally tune three distinct algorithms:
1. **Naive Bayes (`MultinomialNB`)**
   - Tuned Parameter: `alpha` (Additive smoothing)
2. **Logistic Regression**
   - Tuned Parameter: `C` (Inverse of regularization strength)
   - *Note: Configured with `class_weight='balanced'` to offset an imbalanced dataset.*
3. **Support Vector Machine (`LinearSVC`)**
   - Tuned Parameter: `C`
   - *Note: Wrapper in CalibratedClassifierCV to allow the SVM to output probability confidence scores for the UI.*

All trained models, Label Encoders, Vectorizers, and `spaCy` embeddings are serialized (`.pkl`) into the `/models/` directory for high-speed local inference upon Flask startup.

---

## ⚙️ Setup & Installation Instructions

This project runs 100% locally on your machine.

### 1. Clone the Repository
```bash
git clone https://github.com/dhruvil-d/ML_Lab_Project.git
cd ML_Lab_Project
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed, then install the required PIP packages:
```bash
pip install -r requirements.txt
```

### 3. Download the spaCy NLP Model
The semantic search relies on spaCy's Medium-sized English web model for vector mathematics. Download it directly:
```bash
python -m spacy download en_core_web_md
```

### 4. Train the ML Models & Process Embeddings
Before booting the server, you must run the training script. This script will read `data/movies.csv`, run TF-IDF, execute `GridSearchCV` on all 3 ML algorithms, calculate TF-IDF word clouds, and save the compressed `.pkl` files to `/models/`.

```bash
python train_models.py
```
*(Note: Because it is computing cross-validation and generating 800+ 300-dimension vector embeddings, this step may take between 15-45 seconds).*

### 5. Start the Flask Server
Once the `✅ All models saved to models/ directory!` message appears, boot up the web application:
```bash
python app.py
```

### 6. View the Dashboard
Open your favorite web browser and navigate to:
**http://127.0.0.1:5000**
