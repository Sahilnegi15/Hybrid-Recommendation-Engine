from sqlalchemy.orm import Session

from app.database.models import User
from app.database.models import Movie
from app.database.models import Rating
from app.database.models import Recommendation


# -------------------------
# USER
# -------------------------

def create_user(db: Session, name: str):
    user = User(name=name)

    db.add(user)

    db.commit()

    db.refresh(user)

    return user


def get_user(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()


# -------------------------
# MOVIES
# -------------------------

def add_movie(db: Session, title: str, genres: str):
    movie = Movie(
        title=title,
        genres=genres
    )

    db.add(movie)

    db.commit()

    db.refresh(movie)

    return movie


def get_movie(db: Session, movie_id: int):
    return db.query(Movie).filter(Movie.id == movie_id).first()


def get_all_movies(db: Session):
    return db.query(Movie).all()


# -------------------------
# RATINGS
# -------------------------

def add_rating(
        db: Session,
        user_id: int,
        movie_id: int,
        rating: float
):
    rating_obj = Rating(
        user_id=user_id,
        movie_id=movie_id,
        rating=rating
    )

    db.add(rating_obj)

    db.commit()

    db.refresh(rating_obj)

    return rating_obj


def get_user_ratings(
        db: Session,
        user_id: int
):
    return db.query(Rating).filter(
        Rating.user_id == user_id
    ).all()


# -------------------------
# RECOMMENDATIONS
# -------------------------

def save_recommendation(
        db: Session,
        user_id: int,
        movie_id: int,
        score: float
):

    recommendation = Recommendation(
        user_id=user_id,
        movie_id=movie_id,
        score=score
    )

    db.add(recommendation)

    db.commit()

    db.refresh(recommendation)

    return recommendation


def get_recommendations(
        db: Session,
        user_id: int
):

    return db.query(Recommendation).filter(
        Recommendation.user_id == user_id
    ).all()