from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    client_id: str
    client_secret: str
    redirect_uri: str
    app_secret_key: str
    drive_folder_name: str
    colab_url: str
    notebook_id: str
    db_url: str = Field(..., env="DB_URL")  # Render la inyecta directamente
    hash_secret_key: str
    frontend_url: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
