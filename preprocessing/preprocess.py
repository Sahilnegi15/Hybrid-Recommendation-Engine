import os
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder

DATA_DIR = "data/ml-100k"
OUTPUT_DIR = "preprocessing/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load Ratings
# -----------------------------

ratings = pd.read_csv(
    os.path.join(DATA_DIR, "u.data"),
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# -----------------------------
# Load Movies
# -----------------------------

genre_columns = [
    "unknown", "Action", "Adventure", "Animation",
    "Children", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
]

movie_columns = [
    "movie_id",
    "title",
    "release_date",
    "video_release_date",
    "IMDb_URL",
] + genre_columns

movies = pd.read_csv(
    os.path.join(DATA_DIR, "u.item"),
    sep="|",
    names=movie_columns,
    encoding="latin-1"
)

# -----------------------------
# Keep Required Columns
# -----------------------------

movies = movies[
    ["movie_id", "title"] + genre_columns
]

# -----------------------------
# Create Genre String
# -----------------------------

def build_genres(row):
    genres = []

    for g in genre_columns:
        if row[g] == 1:
            genres.append(g)

    return "|".join(genres)

movies["genres"] = movies.apply(build_genres, axis=1)

movies = movies[["movie_id", "title", "genres"]]

# -----------------------------
# Encode User IDs
# -----------------------------

user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

ratings["user_idx"] = user_encoder.fit_transform(
    ratings["user_id"]
)

ratings["movie_idx"] = movie_encoder.fit_transform(
    ratings["movie_id"]
)

# -----------------------------
# Save Encoders
# -----------------------------

joblib.dump(
    user_encoder,
    os.path.join(OUTPUT_DIR, "user_mapping.pkl")
)

joblib.dump(
    movie_encoder,
    os.path.join(OUTPUT_DIR, "movie_mapping.pkl")
)

# -----------------------------
# Interaction Matrix
# -----------------------------

interaction_matrix = ratings.pivot_table(
    index="user_idx",
    columns="movie_idx",
    values="rating",
    fill_value=0
)

interaction_matrix.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "interaction_matrix.csv"
    )
)

# -----------------------------
# Save Processed Ratings
# -----------------------------

ratings.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "ratings.csv"
    ),
    index=False
)

# -----------------------------
# Save Movies
# -----------------------------

movies.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "movies.csv"
    ),
    index=False
)

# -----------------------------
# Movie Features
# -----------------------------

movie_features = movies.copy()

movie_features.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "movie_features.csv"
    ),
    index=False
)

print("=" * 50)
print("Preprocessing Complete")
print("=" * 50)
print(f"Ratings : {len(ratings)}")
print(f"Movies  : {len(movies)}")
print(f"Users   : {ratings.user_idx.nunique()}")
print("=" * 50)