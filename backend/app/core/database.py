from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.base import Base
from app.core.config import settings
import psycopg2
from fastapi import HTTPException
# 🔹 Revisar la URL de conexión

engine = create_engine(settings.db_url, pool_size=10, max_overflow=20)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def create_database():
    """Crea la base de datos si no existe."""
    try:
        conn = psycopg2.connect(
            dbname="postgres",  # Nos conectamos primero a la BD por defecto
            user=settings.db_user,
            password=settings.db_password,
            host=settings.db_host,
            port=settings.db_port
        )
        conn.autocommit = True
        cursor = conn.cursor()

        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{settings.db_name}'")
        exists = cursor.fetchone()

        if not exists:
            cursor.execute(f"CREATE DATABASE {settings.db_name}")

        cursor.close()
        conn.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al crear la base de datos: {e}")

def init_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    db.close()


def get_db():
    """Generador de sesión de SQLAlchemy."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

