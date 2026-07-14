from fastapi import APIRouter
from fastapi import HTTPException

from app.schemas import RatingRequest
from app.services.hybrid import HybridRecommender
from app.services.content_based import ContentBasedRecommender

router = APIRouter(
    prefix="/api",
    tags=["Recommendation"]
)

hybrid_model = HybridRecommender()

content_model = ContentBasedRecommender()
content_model.load_model()


@router.get("/recommend/{user_id}")
def recommend_movies(
    user_id: int,
    reference_movie: int | None = None,
    top_k: int = 10
):
    """
    Personalized recommendations
    """

    try:

        recommendations = hybrid_model.recommend(
            user_id=user_id,
            reference_movie=reference_movie,
            top_k=top_k
        )

        return {
            "user_id": user_id,
            "recommendations": recommendations
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get("/similar/{movie_id}")
def similar_movies(
    movie_id: int,
    top_k: int = 10
):
    """
    Content-based recommendations
    """

    recommendations = content_model.recommend(
        movie_id,
        top_k
    )

    return {
        "movie_id": movie_id,
        "recommendations": recommendations
    }


@router.post("/rating")
def add_rating(
    request: RatingRequest
):

    """
    Placeholder for saving ratings.

    We'll connect PostgreSQL in the next step.
    """

    return {

        "message": "Rating received",

        "user_id": request.user_id,

        "movie_id": request.movie_id,

        "rating": request.rating
    }


@router.get("/health")
def health():

    return {

        "status": "healthy"
    }