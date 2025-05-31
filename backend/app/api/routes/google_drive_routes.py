from fastapi import APIRouter, Request, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import uuid
import os
import json
import tempfile
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from .auth_routes import get_current_user
from app.utils.create_notebook import create_notebook
import anyio
from app.core.database import get_db
from app.crud import user_crud
from sqlalchemy.orm import Session

router = APIRouter()

def get_drive_service(token):
    print("🔑 TOKEN:", token)
    creds = Credentials(
        token=token.get("access_token"),
        refresh_token=token.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
        scopes=["https://www.googleapis.com/auth/drive.file"]
    )
    return build("drive", "v3", credentials=creds)



# Ruta para subir un notebook a Google Drive
@router.post("/upload_notebook")
async def upload_notebook(
    request: Request,
    notebook_path: str = Form(...),
    user: dict = Depends(get_current_user)
):
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
   
    token = request.session.get("token")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No token available",
            headers={"WWW-Authenticate": "Bearer"},
        )
   
    try:
        # Obtener el servicio de Drive
        drive_service = get_drive_service(token)
       
        # Obtener o crear la carpeta principal
        folder_name = os.getenv("DRIVE_FOLDER_NAME", "ColaboAutomation")
        folder_id = await get_or_create_folder(drive_service, folder_name)
       
        # Verificar que el archivo existe
        if not os.path.exists(notebook_path):
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"error": f"Archivo no encontrado: {notebook_path}"}
            )
       
        # Obtener el nombre del archivo
        file_name = os.path.basename(notebook_path)
       
        # Subir el archivo a Google Drive
        file_metadata = {
            'name': file_name,
            'parents': [folder_id]
        }
       
        media = MediaFileUpload(
            notebook_path,
            mimetype='application/x-ipynb+json',
            resumable=True
        )
       
        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, webViewLink'
        ).execute()
       
        # Generar enlace de Colab
        file_id = file.get('id')
        colab_link = f"https://colab.research.google.com/drive/{file_id}"
       
        return {
            "file_id": file_id,
            "colab_link": colab_link,
            "status": "uploaded"
        }
       
    except HttpError as error:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"Error de Google Drive: {str(error)}"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"Error inesperado: {str(e)}"}
        )


# Ruta para listar notebooks en Google Drive
@router.get("/list_notebooks")
async def list_notebooks(
    request: Request,
    user: dict = Depends(get_current_user)
):
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
   
    token = request.session.get("token")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No token available",
            headers={"WWW-Authenticate": "Bearer"},
        )
   
    try:
        # Obtener el servicio de Drive
        drive_service = get_drive_service(token)
       
        # Obtener o crear la carpeta principal
        folder_name = os.getenv("DRIVE_FOLDER_NAME", "ColaboAutomation")
        folder_id = await get_or_create_folder(drive_service, folder_name)
       
        # Listar archivos en la carpeta
        response = drive_service.files().list(
            q=f"'{folder_id}' in parents and mimeType='application/json' and trashed=false",
            spaces='drive',
            fields='files(id, name, webViewLink, createdTime)'
        ).execute()
       
        files = response.get('files', [])
       
        # Convertir a formato de respuesta
        notebooks = []
        for file in files:
            notebooks.append({
                "id": file.get('id'),
                "name": file.get('name'),
                "created_time": file.get('createdTime'),
                "colab_link": f"https://colab.research.google.com/drive/{file.get('id')}"
            })
       
        return {"notebooks": notebooks}
       
    except HttpError as error:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"Error de Google Drive: {str(error)}"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": f"Error inesperado: {str(e)}"}
        )


def get_drive_service_from_refresh(refresh_token: str):
    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET")
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


@router.post("/create_and_upload_notebook/{model_id:path}")
async def create_and_upload_notebook(
    request: Request,
    model_id: str,
    db: Session = Depends(get_db),
    session_user: dict = Depends(get_current_user)
):
    if not session_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # ✅ Buscar el usuario completo en la base de datos
    from app.crud import user_crud
    db_user = user_crud.get_user_by_email(db, session_user["email"])
    if not db_user or not db_user.google_drive_token:
        raise HTTPException(status_code=400, detail="No Google Drive token found")

    try:
        # 🧠 Obtener datos del modelo
        from app.api.routes.hugging_face_routes import classifica_modelo_sync
        model_info = await anyio.to_thread.run_sync(classifica_modelo_sync, model_id)
        print("🧠 Modelo:", model_info)

        # 📓 Crear notebook
        notebook = await anyio.to_thread.run_sync(create_notebook, model_id, model_info)
        if not isinstance(notebook, nbformat.NotebookNode):
            raise TypeError("create_notebook must return a nbformat.NotebookNode")

        # 💾 Guardar temporalmente
        with tempfile.NamedTemporaryFile(suffix='.ipynb', delete=False) as temp_file:
            temp_file.write(nbformat.writes(notebook).encode())
            temp_notebook_path = temp_file.name
        print("📄 Notebook generado:", temp_notebook_path)

        # 🔑 Construir credenciales desde refresh_token guardado
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        creds = Credentials(
            token=None,
            refresh_token=db_user.google_drive_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=os.getenv("CLIENT_ID"),
            client_secret=os.getenv("CLIENT_SECRET")
        )
        drive_service = build("drive", "v3", credentials=creds)

        # 📁 Crear carpeta si no existe
        folder_name = os.getenv("DRIVE_FOLDER_NAME", "ColaboAutomation")
        folder_id = await get_or_create_folder(drive_service, folder_name)

        # ☁️ Subir a Google Drive
        from googleapiclient.http import MediaFileUpload
        file_name = f"{model_id.replace('/', '_')}.ipynb"
        file_metadata = {"name": file_name, "parents": [folder_id]}
        media = MediaFileUpload(temp_notebook_path, mimetype='application/x-ipynb+json', resumable=True)

        file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, webViewLink'
        ).execute()

        file_id = file.get("id")
        colab_link = f"https://colab.research.google.com/drive/{file_id}"
        print("✅ Archivo subido:", colab_link)

        os.unlink(temp_notebook_path)

        return {
            "file_id": file_id,
            "colab_link": colab_link,
            "model_id": model_id,
            "status": "success"
        }

    except HttpError as error:
        print("❌ HttpError:", error)
        return JSONResponse(status_code=500, content={"error": f"Google Drive error: {str(error)}"})

    except Exception as e:
        print("❌ Error inesperado:", e)
        return JSONResponse(status_code=500, content={"error": f"Error inesperado: {str(e)}"})
