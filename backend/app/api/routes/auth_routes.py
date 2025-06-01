from fastapi import APIRouter, Request, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from starlette.middleware.sessions import SessionMiddleware
import os
from typing import Optional
import json
from pathlib import Path
from app.core.config import settings
from app.core.database import get_db
from app.schemas.user_schema import UserCreate
from app.crud import user_crud
from datetime import datetime, timedelta
from jose import jwt
from sqlalchemy.orm import Session
from app.core.auth import create_access_token  # asegúrate de tener esto importado


CLIENT_ID = settings.client_id
CLIENT_SECRET = settings.client_secret
REDIRECT_URI = settings.redirect_uri
SECRET_KEY = settings.hash_secret_key
ALGORITHM = "HS256"


oauth = OAuth()

oauth.register(
    name="google",
    client_id=settings.client_id,
    client_secret=settings.client_secret,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={
        "scope": "openid email profile https://www.googleapis.com/auth/drive.file",
    }
)


router = APIRouter()


# Función para verificar si el usuario está autenticado
async def get_current_user(request: Request) -> Optional[dict]:
    user = request.session.get("user")
    if not user:
        return None
    return user


# Ruta para iniciar el flujo de autenticación
@router.get("/login")
async def login(request: Request):
    redirect_uri = REDIRECT_URI
    return await oauth.google.authorize_redirect(
        request,
        redirect_uri,
        access_type="offline",     # ← sí o sí aquí
        prompt="consent",          # ← y aquí
    )


@router.get("/callback")

async def auth_callback(request: Request, db: Session = Depends(get_db)):
    token = await oauth.google.authorize_access_token(request)
    print("TOKEN DEBUG:", token)
    user_info = token.get("userinfo")

    if not user_info:
        raise HTTPException(status_code=400, detail="No se pudo obtener la información del usuario")

    # 🔐 Extraer tokens de Google
    access_token = token.get("access_token")
    refresh_token = token.get("refresh_token")
    expires_in = token.get("expires_in")  # en segundos

    token_expiry = datetime.utcnow() + timedelta(seconds=expires_in) if expires_in else None

    # 📦 Preparar datos del usuario
    user_data = UserCreate(
        name=user_info.get("name"),
        email=user_info.get("email"),
        google_id=user_info.get("sub"),
        hf_token=None,
        google_drive_token=refresh_token,
    )

    # 🔄 Crear o actualizar usuario en DB
    db_user = user_crud.get_user_by_email(db, user_data.email)
    if db_user:
        db_user.google_drive_token = refresh_token or db_user.google_drive_token
        db_user.google_drive_access_token = access_token
        db_user.google_drive_token_expiry = token_expiry
    else:
        db_user = user_crud.create_user(db, user_data)
        db_user.google_drive_token = refresh_token
        db_user.google_drive_access_token = access_token
        db_user.google_drive_token_expiry = token_expiry

    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # 🔐 Crear JWT interno para autenticar en la app
    jwt_token = create_access_token({"sub": db_user.email})

    # 🔁 Redirigir al frontend con el JWT
    return RedirectResponse(url=f"{settings.frontend_url}/auth/callback?token={jwt_token}")



# Ruta para cerrar sesión
@router.get("/logout")
async def logout(request: Request):
    request.session.pop("user", None)
    request.session.pop("token", None)
    return RedirectResponse(url="/")


# Ruta para obtener información del usuario actual
@router.get("/user")
async def get_user(user: dict = Depends(get_current_user)):
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user




