# app/models/user.py
from sqlalchemy import Column, Integer, String, DateTime
from .base import Base

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True)
    google_id = Column(String, nullable=False, unique=True)
    google_drive_token = Column(String, nullable=True)        # ← Refresh token
    google_drive_access_token = Column(String, nullable=True) # ← Access token
    google_drive_token_expiry = Column(DateTime, nullable=True)