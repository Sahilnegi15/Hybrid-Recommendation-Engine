from app.services.content_based import ContentBasedRecommender

model = ContentBasedRecommender()

model.load_data()

print("Training Content-Based Recommender...")

model.train()

model.save_model()

print("Model Saved!")