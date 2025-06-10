from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from app.api.routes.hugging_face_routes import router as hf_router
from app.api.routes.google_drive_routes import router as google_drive_router
from app.api.routes.auth_routes import router as auth_router
from app.api.routes.user_routes import router as user_router
from app.core.config import settings
from app.core.database import init_db

app = FastAPI(
    title="TFG Notebook Automation",
    description="API para la automatización de notebooks de TFG",
    version="1.0.0"
)

init_db()

app.add_middleware(
    SessionMiddleware, 
    secret_key=settings.app_secret_key,
    https_only=False,  # En local
    same_site="lax",
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://frontend-uwik.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Incluir las rutas
app.include_router(hf_router, prefix="/huggingface", tags=["Hugging Face"])
app.include_router(google_drive_router, prefix="/google_drive", tags=["Google Drive"])
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(user_router, prefix="/users", tags=["Users"])

# Punto de prueba opcional
@app.get("/")
def root():
    return {"message": "API de Modelos LLM desde Hugging Face"}
