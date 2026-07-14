import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:

    def __init__(self):

        self.movies = None

        self.vectorizer = None

        self.similarity_matrix = None

    def load_data(
            self,
            movie_file="preprocessing/processed/movies.csv"
    ):

        self.movies = pd.read_csv(movie_file)

        self.movies["content"] = (
            self.movies["title"].fillna("")
            + " "
            + self.movies["genres"].fillna("")
        )

    def train(self):

        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        tfidf_matrix = self.vectorizer.fit_transform(
            self.movies["content"]
        )

        self.similarity_matrix = cosine_similarity(
            tfidf_matrix,
            tfidf_matrix
        )

    def save_model(
            self,
            path="saved_models/content_based.pkl"
    ):

        os.makedirs("saved_models", exist_ok=True)

        joblib.dump(
            {
                "vectorizer": self.vectorizer,
                "similarity_matrix": self.similarity_matrix,
                "movies": self.movies,
            },
            path,
        )

    def load_model(
            self,
            path="saved_models/content_based.pkl"
    ):

        data = joblib.load(path)

        self.vectorizer = data["vectorizer"]

        self.similarity_matrix = data["similarity_matrix"]

        self.movies = data["movies"]

    def recommend(
            self,
            movie_id,
            top_k=10
    ):

        idx = self.movies.index[
            self.movies["movie_id"] == movie_id
        ]

        if len(idx) == 0:

            return []

        idx = idx[0]

        similarity_scores = list(
            enumerate(self.similarity_matrix[idx])
        )

        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )

        similarity_scores = similarity_scores[1: top_k + 1]

        recommendations = []

        for movie_index, score in similarity_scores:

            movie = self.movies.iloc[movie_index]

            recommendations.append(
                {
                    "movie_id": int(movie["movie_id"]),
                    "title": movie["title"],
                    "genres": movie["genres"],
                    "score": round(float(score), 4),
                }
            )

        return recommendations