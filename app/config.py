from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Health-Insurance-Premium-Prediction"
    version: str = "1.0.0"
    model_path: str = "models/model.pkl"