# ============================================================
# fetch_data.py — Data Collection Pipeline for CineNLP
#
# PURPOSE:
#   Fetches movie data from the OMDB (Open Movie Database) API
#   by searching across many genre-related keywords, retrieves
#   full details for each movie, and saves the result as a JSON
#   dataset at data/movies.json.
#
# USAGE:
#   Run this script ONCE before training:
#       python fetch_data.py
#
# OUTPUT:
#   data/movies.json — array of movie objects with fields:
#     imdbID, title, year, genre, genres_list, plot, poster,
#     rating, director, actors, runtime
# ============================================================

# --- Standard library imports ---
import requests  # HTTP client for making API calls to OMDB
import json      # For reading/writing JSON data files
import time      # For rate-limiting delays between API calls
import os        # For file path construction

# ============================================================
# CONFIGURATION
# ============================================================

# OMDB API key — required for authentication with the API
API_KEY = "b7c63dca"

# Base URL for all OMDB API requests
BASE_URL = "http://www.omdbapi.com/"

# Search terms used to fetch a diverse set of movies across genres.
# Each term is passed as the 's' (search) parameter to OMDB.
# More terms = more diverse dataset, but more API calls.
SEARCH_TERMS = [
    "action", "comedy", "drama", "horror", "thriller",
    "romance", "adventure", "sci-fi", "fantasy", "mystery",
    "animation", "crime", "war", "family", "musical",
    "superhero", "space", "love", "fight", "magic",
    "zombie", "robot", "spy", "king", "hero",
    "dark", "night", "star", "fire", "world",
    "iron", "fast", "mission", "game", "death",
    "life", "dream", "time", "monster", "ghost",
    "shark", "dragon", "hunter", "battle", "legend",
    "planet", "alien", "warrior", "jungle", "ocean",
    # Additional terms for more data coverage
    "detective", "police", "murder", "revenge", "escape",
    "princess", "wizard", "pirate", "treasure", "school",
    "christmas", "wedding", "dance", "music", "race",
    "secret", "island", "castle", "army", "survival",
    "vampire", "witch", "angel", "devil", "curse",
    "heist", "prison", "court", "serial", "hunt",
    "storm", "ice", "snow", "mountain", "desert",
    "future", "ancient", "ninja", "samurai", "gladiator",
]

# Number of result pages to fetch per search term (each page = 10 results from OMDB)
PAGES_PER_TERM = 2


# ============================================================
# STEP 1: Fetch movies from the OMDB API
# ============================================================

def fetch_movies():
    """
    Fetch movies from OMDB using various search terms.

    Strategy:
      1. Load any existing data/movies.json to avoid re-fetching duplicates
      2. For each search term × page, call the OMDB search endpoint
      3. For each search result, call the OMDB detail endpoint (full plot)
      4. Filter out movies with missing/short plots or missing genres
      5. Return the full list of unique movies (keyed by imdbID)
    """

    # --- Load existing data to avoid re-fetching ---
    data_path = os.path.join(os.path.dirname(__file__), "data", "movies.json")
    all_movies = {}  # Dict keyed by imdbID for deduplication

    if os.path.exists(data_path):
        # Read previously fetched movies and index by IMDB ID
        with open(data_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for m in existing:
            all_movies[m["imdbID"]] = m
        print(f"Loaded {len(all_movies)} existing movies")

    # --- Iterate over every search term and page ---
    for term in SEARCH_TERMS:
        for page in range(1, PAGES_PER_TERM + 1):
            print(f"Searching: {term} (page {page})")
            try:
                # OMDB Search endpoint — returns a list of up to 10 matching movies
                resp = requests.get(BASE_URL, params={
                    "apikey": API_KEY,
                    "s": term,          # Search keyword
                    "type": "movie",    # Only movies (not series/episodes)
                    "page": page        # Pagination
                }, timeout=10)
                data = resp.json()

                # Check if the API returned valid results
                if data.get("Response") == "True":
                    for movie in data.get("Search", []):
                        imdb_id = movie.get("imdbID")

                        # Skip if we already have this movie
                        if imdb_id and imdb_id not in all_movies:

                            # OMDB Detail endpoint — fetch full movie info by IMDB ID
                            detail_resp = requests.get(BASE_URL, params={
                                "apikey": API_KEY,
                                "i": imdb_id,       # Lookup by IMDB ID
                                "plot": "full"       # Request the full-length plot
                            }, timeout=10)
                            detail = detail_resp.json()

                            if detail.get("Response") == "True":
                                plot = detail.get("Plot", "N/A")
                                genre = detail.get("Genre", "N/A")

                                # Quality filter: skip movies with no plot, no genre,
                                # or plots shorter than 50 characters (too brief for NLP)
                                if plot != "N/A" and genre != "N/A" and len(plot) > 50:
                                    # Build a clean movie record
                                    all_movies[imdb_id] = {
                                        "imdbID": imdb_id,
                                        "title": detail.get("Title", ""),
                                        "year": detail.get("Year", ""),
                                        "genre": genre,
                                        # Split "Action, Drama, Thriller" → ["Action", "Drama", "Thriller"]
                                        "genres_list": [g.strip() for g in genre.split(",")],
                                        "plot": plot,
                                        "poster": detail.get("Poster", ""),
                                        "rating": detail.get("imdbRating", "N/A"),
                                        "director": detail.get("Director", ""),
                                        "actors": detail.get("Actors", ""),
                                        "runtime": detail.get("Runtime", ""),
                                    }
                                    print(f"  + {detail.get('Title')} ({genre})")

                            # Rate-limit between detail requests (0.15s)
                            time.sleep(0.15)

                # Rate-limit between search requests (0.2s)
                time.sleep(0.2)

            except Exception as e:
                # Log the error but continue with the next term/page
                print(f"  Error: {e}")
                continue

    # Convert the deduplication dict back to a list
    return list(all_movies.values())


# ============================================================
# STEP 2: Main entry point — orchestrate fetch and save
# ============================================================

def main():
    print("=" * 60)
    print("Fetching movies from OMDB API...")
    print("=" * 60)

    # Fetch all movies from the API
    movies = fetch_movies()

    # Save the complete dataset as a JSON file
    output_path = os.path.join(os.path.dirname(__file__), "data", "movies.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(movies, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Saved {len(movies)} movies to {output_path}")

    # --- Print genre distribution summary ---
    # Useful for understanding dataset balance before training
    from collections import Counter
    genre_counts = Counter()
    for m in movies:
        for g in m["genres_list"]:
            genre_counts[g] += 1

    print("\nGenre distribution:")
    for genre, count in genre_counts.most_common(20):
        print(f"  {genre}: {count}")


# Standard Python idiom: only run main() when this file is executed directly
# (not when imported as a module)
if __name__ == "__main__":
    main()
