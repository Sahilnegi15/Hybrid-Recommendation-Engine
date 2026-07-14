from pydantic import BaseModel


class RatingRequest(BaseModel):
    user_id: int
    movie_id: int
    rating: float


class RecommendationResponse(BaseModel):
    movie_id: int
    title: str
    genres: str
    collaborative: float
    content: float
    score: float