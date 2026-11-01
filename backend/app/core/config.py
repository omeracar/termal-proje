from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Termal Kamera Analiz Sistemi"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    
    DATABASE_URL: str = "postgresql://termal_user:termal_pass@localhost:5432/termal_db"
    
    UPLOAD_DIR: str = "data/raw"
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100 MB

    class Config:
        env_file = ".env"

settings = Settings()
