from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

# Carga variables del archivo .env
load_dotenv()

class Settings(BaseSettings):
    client_id: str
    client_secret: str
    redirect_uri: str
    app_secret_key: str
    drive_folder_name: str
    colab_url: str
    notebook_id: str
    api_url: str
    db_user: str
    db_password: str
    db_host: str
    db_port: int
    db_name: str  
    hash_secret_key: str
    frontend_url: str

    @property
    def db_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Instancia global
settings = Settings()
