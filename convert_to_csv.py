import json
import csv
import os

data_path = os.path.join("data", "movies.json")
csv_path = os.path.join("data", "movies.csv")

with open(data_path, "r", encoding="utf-8") as f:
    movies = json.load(f)

if movies:
    keys = movies[0].keys()
    with open(csv_path, "w", encoding="utf-8", newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        for m in movies:
            m_copy = m.copy()
            if isinstance(m_copy.get("genres_list"), list):
                m_copy["genres_list"] = "|".join(m_copy["genres_list"])
            dict_writer.writerow(m_copy)

print(f"Successfully converted {len(movies)} movies to {csv_path}")
