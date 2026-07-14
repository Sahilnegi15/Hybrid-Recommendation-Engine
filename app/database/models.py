from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.database.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String, nullable=False)

    ratings = relationship("Rating", back_populates="user")


class Movie(Base):
    __tablename__ = "movies"

    id = Column(Integer, primary_key=True, index=True)

    title = Column(String, nullable=False)

    genres = Column(String)

    ratings = relationship("Rating", back_populates="movie")


class Rating(Base):
    __tablename__ = "ratings"

    id = Column(Integer, primary_key=True)

    user_id = Column(Integer, ForeignKey("users.id"))

    movie_id = Column(Integer, ForeignKey("movies.id"))

    rating = Column(Float)

    timestamp = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="ratings")

    movie = relationship("Movie", back_populates="ratings")


class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True)

    user_id = Column(Integer)

    movie_id = Column(Integer)

    score = Column(Float)

    created_at = Column(DateTime(timezone=True), server_default=func.now())