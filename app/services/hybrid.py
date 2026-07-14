import pandas as pd

from app.services.collaborative import CollaborativeFiltering
from app.services.content_based import ContentBasedRecommender


class HybridRecommender:

    def __init__(self):

        self.cf = CollaborativeFiltering()

        self.cb = ContentBasedRecommender()

        self.movies = pd.read_csv(
            "preprocessing/processed/movies.csv"
        )

        ratings = self.cf.load_data()

        self.cf.load(ratings)

        self.cb.load_model()

    def recommend(
        self,
        user_id,
        reference_movie=None,
        top_k=10,
        alpha=0.7
    ):
        """
        alpha:
            collaborative weight

        (1-alpha):
            content weight
        """

        collaborative_scores = {}

        cf_predictions = self.cf.recommend(
            user_id,
            top_k=500
        )

        for movie_id, score in cf_predictions:

            collaborative_scores[movie_id] = score

        content_scores = {}

        if reference_movie is not None:

            recommendations = self.cb.recommend(
                reference_movie,
                top_k=500
            )

            for movie in recommendations:

                content_scores[
                    movie["movie_id"]
                ] = movie["score"]

        results = []

        movie_ids = self.movies.movie_id.values

        for movie_id in movie_ids:

            cf_score = collaborative_scores.get(
                movie_id,
                0
            )

            cb_score = content_scores.get(
                movie_id,
                0
            )

            final_score = (
                alpha * cf_score
                +
                (1 - alpha) * cb_score
            )

            movie = self.movies[
                self.movies.movie_id == movie_id
            ].iloc[0]

            results.append(
                {
                    "movie_id": int(movie_id),
                    "title": movie.title,
                    "genres": movie.genres,
                    "collaborative": round(cf_score, 4),
                    "content": round(cb_score, 4),
                    "score": round(final_score, 4)
                }
            )

        results = sorted(
            results,
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:top_k]