# app/utils/google_drive.py
from datetime import datetime, timedelta
import os, httpx
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from sqlalchemy.orm import Session
from app.models.user import User

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")

def refresh_google_access_token(user: User, db: Session) -> str:
    """Renueva el access_token si hace falta y lo guarda."""
    print(f"🔁 Refrescando token de acceso para: {user.email}")
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": user.google_drive_token,
    }
    r = httpx.post(GOOGLE_TOKEN_URL, data=data, timeout=10)
    r.raise_for_status()
    payload = r.json()

    user.google_drive_access_token = payload["access_token"]
    user.google_drive_token_expiry = datetime.utcnow() + timedelta(seconds=payload["expires_in"])
    db.add(user)
    db.commit()
    db.refresh(user)
    return user.google_drive_access_token

def get_drive_service_for_user(user: User, db: Session):
    """Devuelve un `drive_service` listo, renovando token si está expirado."""
    if not user.google_drive_token:
        raise RuntimeError(f"Usuario {user.email} no tiene refresh_token de Google Drive")

    if (not user.google_drive_access_token) or (
        user.google_drive_token_expiry and user.google_drive_token_expiry <= datetime.utcnow()
    ):
        access_token = refresh_google_access_token(user, db)
    else:
        access_token = user.google_drive_access_token

    creds = Credentials(
        token=access_token,
        refresh_token=user.google_drive_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        scopes=["https://www.googleapis.com/auth/drive.file"],
    )
    return build("drive", "v3", credentials=creds)

async def get_or_create_folder(drive_service, folder_name: str):
    response = drive_service.files().list(
        q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
        spaces='drive',
        fields='files(id, name)'
    ).execute()

    if response.get('files'):
        return response['files'][0]['id']

    folder_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }

    folder = drive_service.files().create(
        body=folder_metadata,
        fields='id'
    ).execute()

    return folder.get('id')
