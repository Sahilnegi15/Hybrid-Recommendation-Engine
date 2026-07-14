from fastapi import FastAPI

from app.api.recommendation import router
from app.database.db import Base
from app.database.db import engine

Base.metadata.create_all(bind=engine)

app = FastAPI(

    title="Hybrid Recommendation Engine",

    version="1.0.0",

    description="Hybrid Recommendation API using PyTorch + FastAPI"
)

app.include_router(router)


@app.get("/")
def home():

    return {

        "message": "Hybrid Recommendation Engine",

        "docs": "/docs"
    }