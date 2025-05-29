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

CLIENT_ID = settings.client_id
CLIENT_SECRET = settings.client_secret
REDIRECT_URI = settings.redirect_uri


# Configurar OAuth
oauth = OAuth()
oauth.register(
    name="google",
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={
        "scope": "openid email profile https://www.googleapis.com/auth/drive.file"
    },
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
    return await oauth.google.authorize_redirect(request, redirect_uri)


# Ruta para manejar el callback de Google OAuth
@router.get("/callback")
async def auth_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    user_info = token.get("userinfo")
    if user_info:
        # Guardar información del usuario y token en la sesión
        request.session["user"] = dict(user_info)
        request.session["token"] = token
        return RedirectResponse(url="/")
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


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




