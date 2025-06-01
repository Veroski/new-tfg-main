# app/schemas/user_schema.py

from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    name: str
    email: str
    google_id: str
    google_drive_token: Optional[str] = None
    google_drive_access_token: Optional[str] = None
    google_drive_token_expiry: Optional[datetime] = None

class UserCreate(UserBase):
    pass

class UserUpdate(BaseModel):
    name: Optional[str]
    email: Optional[str]
    google_id: Optional[str]
    google_drive_token: Optional[str]
    google_drive_access_token: Optional[str]
    google_drive_token_expiry: Optional[datetime]

class UserOut(UserBase):
    id: int
    class Config:
        orm_mode = True
