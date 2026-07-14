import mlflow
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from app.services.collaborative import CollaborativeFiltering


cf = CollaborativeFiltering()

ratings = cf.load_data()

cf.build_model(ratings)

device = cf.device

users = torch.tensor(
    ratings.user_idx.values,
    dtype=torch.long
)

movies = torch.tensor(
    ratings.movie_idx.values,
    dtype=torch.long
)

targets = torch.tensor(
    ratings.rating.values,
    dtype=torch.float32
)

dataset = TensorDataset(
    users,
    movies,
    targets
)

loader = DataLoader(
    dataset,
    batch_size=512,
    shuffle=True
)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    cf.model.parameters(),
    lr=0.001
)

epochs = 10

mlflow.set_experiment(
    "Hybrid Recommendation Engine"
)

with mlflow.start_run():

    mlflow.log_param(
        "epochs",
        epochs
    )

    mlflow.log_param(
        "batch_size",
        512
    )

    mlflow.log_param(
        "embedding_dim",
        64
    )

    for epoch in range(epochs):

        total_loss = 0

        cf.model.train()

        for users, movies, ratings in loader:

            users = users.to(device)
            movies = movies.to(device)
            ratings = ratings.to(device)

            optimizer.zero_grad()

            predictions = cf.model(
                users,
                movies
            )

            loss = criterion(
                predictions,
                ratings
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        mlflow.log_metric(
            "loss",
            avg_loss,
            step=epoch
        )

        print(
            f"Epoch {epoch+1}/{epochs}"
            f" Loss={avg_loss:.4f}"
        )

    cf.save()

    mlflow.pytorch.log_model(
        cf.model,
        "ncf_model"
    )

print("Training Complete")