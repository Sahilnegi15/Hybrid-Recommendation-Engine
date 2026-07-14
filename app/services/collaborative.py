import os
import joblib
import torch
import torch.nn as nn
import pandas as pd


class NCF(nn.Module):

    def __init__(
        self,
        num_users,
        num_movies,
        embedding_dim=64
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim
        )

        self.movie_embedding = nn.Embedding(
            num_movies,
            embedding_dim
        )

        self.network = nn.Sequential(

            nn.Linear(
                embedding_dim * 2,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                1
            )
        )

    def forward(
        self,
        users,
        movies
    ):

        user_vector = self.user_embedding(users)

        movie_vector = self.movie_embedding(movies)

        x = torch.cat(
            [
                user_vector,
                movie_vector
            ],
            dim=1
        )

        rating = self.network(x)

        return rating.squeeze()


class CollaborativeFiltering:

    def __init__(self):

        self.model = None

        self.user_encoder = None

        self.movie_encoder = None

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    def load_data(self):

        ratings = pd.read_csv(
            "preprocessing/processed/ratings.csv"
        )

        self.user_encoder = joblib.load(
            "preprocessing/processed/user_mapping.pkl"
        )

        self.movie_encoder = joblib.load(
            "preprocessing/processed/movie_mapping.pkl"
        )

        return ratings

    def build_model(
        self,
        ratings
    ):

        num_users = ratings.user_idx.nunique()

        num_movies = ratings.movie_idx.nunique()

        self.model = NCF(
            num_users,
            num_movies
        ).to(self.device)

    def save(self):

        os.makedirs(
            "saved_models",
            exist_ok=True
        )

        torch.save(
            self.model.state_dict(),
            "saved_models/collaborative.pt"
        )

    def load(self, ratings):

        self.build_model(ratings)

        self.model.load_state_dict(
            torch.load(
                "saved_models/collaborative.pt",
                map_location=self.device
            )
        )

        self.model.eval()

    def predict(
        self,
        user_id,
        movie_id
    ):

        user = self.user_encoder.transform(
            [user_id]
        )[0]

        movie = self.movie_encoder.transform(
            [movie_id]
        )[0]

        user = torch.tensor(
            [user],
            dtype=torch.long
        ).to(self.device)

        movie = torch.tensor(
            [movie],
            dtype=torch.long
        ).to(self.device)

        with torch.no_grad():

            prediction = self.model(
                user,
                movie
            )

        return float(prediction)
    
    