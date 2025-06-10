# app/api/routes/google_drive_routes.py
from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import os, tempfile, nbformat, anyio
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

from app.core.auth import get_current_user     # JWT → User ORM
from app.utils.google_drive_utils import get_drive_service_for_user, get_or_create_folder
from app.utils.create_notebook import create_notebook
from app.core.database import get_db
from app.crud import user_crud

router = APIRouter()

# ▶ Upload notebook
@router.post("/upload_notebook")
async def upload_notebook(
    notebook_path: str = Form(...),
    db: Session = Depends(get_db),
    user = Depends(get_current_user)
):
    if not os.path.exists(notebook_path):
        raise HTTPException(404, f"Archivo no encontrado: {notebook_path}")

    try:
        drive = get_drive_service_for_user(user, db)
        folder_id = await get_or_create_folder(
            drive, os.getenv("DRIVE_FOLDER_NAME", "ColaboAutomation")
        )

        file_name = os.path.basename(notebook_path)
        media = MediaFileUpload(notebook_path, mimetype="application/x-ipynb+json", resumable=True)
        meta  = {"name": file_name, "parents": [folder_id]}

        file  = drive.files().create(body=meta, media_body=media, fields="id, webViewLink").execute()
        return {
            "file_id": file["id"],
            "colab_link": f"https://colab.research.google.com/drive/{file['id']}",
            "status": "uploaded",
        }

    except HttpError as e:
        raise HTTPException(500, f"Google Drive error: {e}")


# ▶ List notebooks
@router.get("/list_notebooks")
async def list_notebooks(db: Session = Depends(get_db), user = Depends(get_current_user)):
    drive = get_drive_service_for_user(user, db)
    folder_id = await get_or_create_folder(drive, os.getenv("DRIVE_FOLDER_NAME", "ColaboAutomation"))

    res = drive.files().list(
        q=f"'{folder_id}' in parents and mimeType='application/x-ipynb+json' and trashed=false",
        spaces="drive",
        fields="files(id, name, webViewLink, createdTime)",
    ).execute()

    return {
        "notebooks": [
            {
                "id": f["id"],
                "name": f["name"],
                "created_time": f["createdTime"],
                "colab_link": f"https://colab.research.google.com/drive/{f['id']}",
            }
            for f in res.get("files", [])
        ]
    }


# ▶ Create and upload notebook
@router.post("/create_and_upload_notebook/{model_id:path}")
async def create_and_upload_notebook(
    request: Request,
    model_id: str,
    selected_weight_file: str = None,  # <--- NUEVO
    db: Session = Depends(get_db),
    session_user: dict = Depends(get_current_user)
):
    if not session_user:
        raise HTTPException(status_code=401, detail="Not authenticated")

    db_user = session_user
    if not db_user or not db_user.google_drive_token:
        raise HTTPException(status_code=400, detail="No Google Drive token found")

    try:
        from app.api.routes.hugging_face_routes import classifica_modelo_sync
        model_info = await anyio.to_thread.run_sync(classifica_modelo_sync, model_id, session_user.hf_token, selected_weight_file)

        notebook = await anyio.to_thread.run_sync(lambda: create_notebook(model_id, model_info, user=db_user))
        if not isinstance(notebook, nbformat.NotebookNode):
            raise TypeError("create_notebook must return a nbformat.NotebookNode")

        with tempfile.NamedTemporaryFile(suffix='.ipynb', delete=False) as temp_file:
            temp_file.write(nbformat.writes(notebook).encode())
            temp_notebook_path = temp_file.name

        drive_service = get_drive_service_for_user(db_user, db)
        folder_name = os.getenv("DRIVE_FOLDER_NAME", "ColaboAutomation")
        folder_id = await get_or_create_folder(drive_service, folder_name)

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
        os.unlink(temp_notebook_path)

        return {
            "file_id": file_id,
            "colab_link": colab_link,
            "model_id": model_id,
            "status": "success"
        }

    except HttpError as error:
        return JSONResponse(status_code=500, content={"error": f"Google Drive error: {str(error)}"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error inesperado: {str(e)}"})