"""
fetch_data.py — Fetches movies from OMDB API and builds a training dataset.
Run this ONCE to generate data/movies.json.
"""

import requests
import json
import time
import os

API_KEY = "b7c63dca"
BASE_URL = "http://www.omdbapi.com/"

# Search terms to fetch diverse movies across genres
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
    # Additional terms for more data
    "detective", "police", "murder", "revenge", "escape",
    "princess", "wizard", "pirate", "treasure", "school",
    "christmas", "wedding", "dance", "music", "race",
    "secret", "island", "castle", "army", "survival",
    "vampire", "witch", "angel", "devil", "curse",
    "heist", "prison", "court", "serial", "hunt",
    "storm", "ice", "snow", "mountain", "desert",
    "future", "ancient", "ninja", "samurai", "gladiator",
]
PAGES_PER_TERM = 2  # Fetch 2 pages per search term


def fetch_movies():
    """Fetch movies from OMDB using various search terms."""
    # Load existing data to avoid re-fetching
    data_path = os.path.join(os.path.dirname(__file__), "data", "movies.json")
    all_movies = {}
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for m in existing:
            all_movies[m["imdbID"]] = m
        print(f"Loaded {len(all_movies)} existing movies")
    
    for term in SEARCH_TERMS:
        for page in range(1, PAGES_PER_TERM + 1):
            print(f"Searching: {term} (page {page})")
            try:
                resp = requests.get(BASE_URL, params={
                    "apikey": API_KEY,
                    "s": term,
                    "type": "movie",
                    "page": page
                }, timeout=10)
                data = resp.json()
                
                if data.get("Response") == "True":
                    for movie in data.get("Search", []):
                        imdb_id = movie.get("imdbID")
                        if imdb_id and imdb_id not in all_movies:
                            detail_resp = requests.get(BASE_URL, params={
                                "apikey": API_KEY,
                                "i": imdb_id,
                                "plot": "full"
                            }, timeout=10)
                            detail = detail_resp.json()
                            
                            if detail.get("Response") == "True":
                                plot = detail.get("Plot", "N/A")
                                genre = detail.get("Genre", "N/A")
                                
                                if plot != "N/A" and genre != "N/A" and len(plot) > 50:
                                    all_movies[imdb_id] = {
                                        "imdbID": imdb_id,
                                        "title": detail.get("Title", ""),
                                        "year": detail.get("Year", ""),
                                        "genre": genre,
                                        "genres_list": [g.strip() for g in genre.split(",")],
                                        "plot": plot,
                                        "poster": detail.get("Poster", ""),
                                        "rating": detail.get("imdbRating", "N/A"),
                                        "director": detail.get("Director", ""),
                                        "actors": detail.get("Actors", ""),
                                        "runtime": detail.get("Runtime", ""),
                                    }
                                    print(f"  + {detail.get('Title')} ({genre})")
                            
                            time.sleep(0.15)
                
                time.sleep(0.2)
            except Exception as e:
                print(f"  Error: {e}")
                continue
    
    return list(all_movies.values())


def main():
    print("=" * 60)
    print("Fetching movies from OMDB API...")
    print("=" * 60)
    
    movies = fetch_movies()
    
    # Save dataset
    output_path = os.path.join(os.path.dirname(__file__), "data", "movies.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(movies, f, indent=2, ensure_ascii=False)
    
    print(f"\nDone! Saved {len(movies)} movies to {output_path}")
    
    # Print genre distribution
    from collections import Counter
    genre_counts = Counter()
    for m in movies:
        for g in m["genres_list"]:
            genre_counts[g] += 1
    
    print("\nGenre distribution:")
    for genre, count in genre_counts.most_common(20):
        print(f"  {genre}: {count}")


if __name__ == "__main__":
    main()
