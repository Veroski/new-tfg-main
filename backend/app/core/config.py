from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

# Carga variables del archivo .env
load_dotenv()

class Settings(BaseSettings):
    token: str
    client_id: str
    client_secret: str
    redirect_uri: str
    app_secret_key: str
    drive_folder_name: str
    colab_url: str
    notebook_id: str
    api_url: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Instancia global
settings = Settings()
