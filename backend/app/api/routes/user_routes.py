# app/routes/user_routes.py
from fastapi import APIRouter, Depends, Form, HTTPException
from sqlalchemy.orm import Session
from app.schemas.user_schema import UserCreate, UserUpdate, UserOut
from app.crud import user_crud
from app.core.database import get_db  # Asegúrate de tener esto definido
from app.models.user import User
from app.core.auth import get_current_user  # Asegúrate de tener esto definido

router = APIRouter()

@router.get("/me", response_model=UserOut)
def read_current_user(current_user: User = Depends(get_current_user)):
    return current_user

@router.get("/{user_id}", response_model=UserOut)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = user_crud.get_user(db, user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@router.get("/", response_model=list[UserOut])
def list_users(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    return user_crud.get_users(db, skip=skip, limit=limit)

@router.put("/{user_id}", response_model=UserOut)
def update_user(user_id: int, user: UserUpdate, db: Session = Depends(get_db)):
    db_user = user_crud.update_user(db, user_id, user)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user

@router.delete("/{user_id}", response_model=UserOut)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    db_user = user_crud.delete_user(db, user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@router.get("/me/hf-token", response_model=str | None)
def get_my_hf_token(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    🔐 Devuelve el Hugging Face token del usuario actual.
    """
    return user_crud.get_user_hf_token(db, user_id=current_user.id)


@router.post("/me/hf-token", response_model=str)
@router.put("/me/hf-token", response_model=str)
def set_my_hf_token(
    hf_token: str = Form(...),                     # 👈🏻  AHORA ES Form,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    🔐 Guarda o actualiza el Hugging Face token del usuario actual.
    """
    updated_user = user_crud.update_user_hf_token(db, user_id=current_user.id, hf_token=hf_token)
    if not updated_user:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return updated_user.hf_token
