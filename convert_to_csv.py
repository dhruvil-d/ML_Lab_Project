# ============================================================
# convert_to_csv.py — JSON-to-CSV Converter for CineNLP
#
# PURPOSE:
#   Converts the raw data/movies.json dataset into data/movies.csv
#   format. The CSV uses pipe-delimited values for the genres_list
#   column (e.g., "Action|Drama|Thriller") since CSV cells can't
#   natively hold Python lists.
#
# USAGE:
#   python convert_to_csv.py
#
# INPUT:   data/movies.json
# OUTPUT:  data/movies.csv
# ============================================================

import json  # For parsing the JSON input file
import csv   # For writing the CSV output file
import os    # For constructing file paths

# --- Build file paths relative to the working directory ---
data_path = os.path.join("data", "movies.json")   # Input JSON file
csv_path = os.path.join("data", "movies.csv")      # Output CSV file

# --- Load the JSON dataset ---
# movies.json contains an array of movie objects
with open(data_path, "r", encoding="utf-8") as f:
    movies = json.load(f)  # Parse into a list of dicts

# --- Convert and write to CSV ---
if movies:
    # Use the keys from the first movie object as the CSV column headers
    keys = movies[0].keys()

    # Open the output CSV file with utf-8 encoding and no extra blank lines (newline='')
    with open(csv_path, "w", encoding="utf-8", newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)

        # Write the header row (column names)
        dict_writer.writeheader()

        # Write each movie as a CSV row
        for m in movies:
            m_copy = m.copy()  # Work on a copy to avoid mutating the original

            # Convert the genres_list from a Python list ["Action", "Drama"]
            # to a pipe-delimited string "Action|Drama" for CSV storage,
            # since CSV cells are flat strings and can't hold lists.
            if isinstance(m_copy.get("genres_list"), list):
                m_copy["genres_list"] = "|".join(m_copy["genres_list"])

            dict_writer.writerow(m_copy)

# Print confirmation with the total number of movies converted
print(f"Successfully converted {len(movies)} movies to {csv_path}")
