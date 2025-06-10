# app/crud/crud_user.py
from sqlalchemy.orm import Session
from app.models.user import User
from app.schemas.user_schema import UserCreate, UserUpdate

def get_user(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    return db.query(User).offset(skip).limit(limit).all()

def create_user(db: Session, user: UserCreate):
    db_user = User(
        name=user.name,
        email=user.email,
        google_id=user.google_id,
        google_drive_token=user.google_drive_token,
        google_drive_access_token=user.google_drive_access_token,
        google_drive_token_expiry=user.google_drive_token_expiry
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def update_user(db: Session, user_id: int, user_update: UserUpdate):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return None
    for key, value in user_update.dict(exclude_unset=True).items():
        setattr(user, key, value)
    db.commit()
    db.refresh(user)
    return user

def delete_user(db: Session, user_id: int):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return None
    db.delete(user)
    db.commit()
    return user

def get_user_hf_token(db: Session, user_id: int):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return None
    return user.hf_token

def update_user_hf_token(db: Session, user_id: int, hf_token: str):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return None
    user.hf_token = hf_token
    db.commit()
    db.refresh(user)
    return user

def post_user_hf_token(db: Session, user_id: int, hf_token: str):
    """
    This function is used to set or update the Hugging Face token for a user.
    It is a convenience function that combines the retrieval and update into one step.
    """
    return update_user_hf_token(db, user_id, hf_token)