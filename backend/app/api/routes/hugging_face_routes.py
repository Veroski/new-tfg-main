import json
import anyio
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Any, List, Optional, Dict
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from app.core.auth import get_current_user
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub import hf_hub_download

import requests
from app.utils import model_helper_updated as model_helper
from app.utils.create_notebook import create_notebook
import nbformat as nbf
from io import BytesIO
from starlette.responses import StreamingResponse, Response
from app.core.config import settings
from app.crud.user_crud import get_user_hf_token
from app.api.helpers import (
    es_archivo_modelo,
    es_tag_valido,
    analizar_calidad,
    evaluar_colab,
    SHARD_RGX,
    nombre_grupo_shards,
    PIPELINES,
    EXT_MODELO,
)

router = APIRouter()
api    = HfApi()


# ---------- /detalles_modelo/{id} ----------------------------
@router.get("/detalles_modelo/{model_id:path}")
def obtener_tamanos_archivos(model_id: str, current_user: dict = Depends(get_current_user)):
    info = api.model_info(repo_id=model_id, token=current_user.hf_token, files_metadata=True)

    grupos: Dict[str, list] = {}
    individuales = []

    for meta in info.siblings:
        nombre = meta.rfilename
        if not es_archivo_modelo(nombre):
            continue

        m = SHARD_RGX.match(nombre)
        if m:
            prefix = m.group(1)
            grupos.setdefault(prefix, []).append(meta)
        else:
            individuales.append(meta)

    archivos = []

    # Procesamos grupos de shards
    for prefix, files in grupos.items():
        total_bytes = sum(f.size or 0 for f in files)
        total_gb = round(total_bytes / (1024 ** 3), 2) if total_bytes else None

        variante, rank, nota = analizar_calidad(prefix, total_bytes)
        colab_flag, colab_msg = evaluar_colab(total_gb, rank)

        archivos.append(
            {
                "archivo": nombre_grupo_shards(prefix, len(files)),
                "tamaño_bytes": total_bytes,
                "tamaño_gb": total_gb,
                "variant": variante,
                "rank": rank,
                "recomendacion": nota,
                "colab_status": colab_flag,
                "colab_msg": colab_msg,
            }
        )

    # Procesamos archivos individuales
    for meta in individuales:
        nombre = meta.rfilename
        size_b = meta.size or 0
        size_g = round(size_b / (1024 ** 3), 2) if size_b else None

        variante, rank, nota = analizar_calidad(nombre, size_b)
        colab_flag, colab_msg = evaluar_colab(size_g, rank)

        archivos.append(
            {
                "archivo": nombre,
                "tamaño_bytes": size_b,
                "tamaño_gb": size_g,
                "variant": variante,
                "rank": rank,
                "recomendacion": nota,
                "colab_status": colab_flag,
                "colab_msg": colab_msg,
            }
        )

    archivos.sort(
        key=lambda x: (
            {"✅": 0, "⚠️": 1, "❌": 2}.get(x["colab_status"], 3),
            -x["rank"],
            x["tamaño_bytes"] or 0,
        )
    )

    return JSONResponse(content=jsonable_encoder(archivos))


# ---------- /obtener_readme/{id} ----------------------------
@router.get("/obtener_readme/{model_id:path}")
async def obtener_readme(model_id: str):
    """
    Obtiene el contenido del README.md de un modelo de Hugging Face.
    
    Args:
        model_id: ID del modelo en Hugging Face
        
    Returns:
        Contenido del README.md como texto plano
    """
    try:
        # Intentamos descargar el README.md
        readme_path = await anyio.to_thread.run_sync(
            hf_hub_download, 
            model_id, 
            "README.md", 
        )
        
        # Leemos el contenido del archivo
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
            
        return Response(content=readme_content, media_type="text/markdown")
    except Exception as e:
        # Si hay algún error, devolvemos un mensaje de error
        return JSONResponse(
            status_code=404,
            content={"error": f"No se pudo obtener el README.md: {str(e)}"}
        )

# ---- /route_model/{id} ---------------------------------
@router.get("/route_model/{model_id:path}")
async def route_model(model_id: str):
    """
    Ruta principal para generar un notebook para un modelo.
    Devuelve un JSON con la URL del notebook generado.
    """
    try:
        # Obtener información del modelo
        
        # Generar URL para descargar el notebook
        notebook_url = f"/huggingface/genera_notebook/{model_id}"
        
        return JSONResponse(content={
            "status": "success",
            "model_id": model_id,
            "notebook": notebook_url
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error al generar notebook: {str(e)}"}
        )



# ---- /genera_notebook -----------------------------------
@router.get("/genera_notebook/{model_id:path}")
async def genera_notebook(
    model_id: str, 
    archivo: Optional[str] = Query(None),
    session_user: dict = Depends(get_current_user)
):
    """Genera el .ipynb en segundo plano y lo devuelve como descarga."""
    info = await anyio.to_thread.run_sync(classifica_modelo_sync, model_id, session_user.hf_token, archivo)
    
    # Si se especifica un archivo, añadirlo a la información del modelo
    if archivo:
        info["selected_weight_file"] = archivo
        # Reprocesar la información con el archivo seleccionado
        info = model_helper.process_model_info(info)
    
    # Usar la función create_notebook en lugar de model_helper.build_notebook
    nb = await anyio.to_thread.run_sync(create_notebook, model_id, info, user=session_user)

    data = await anyio.to_thread.run_sync(nbf.writes, nb)
    buffer = BytesIO(data.encode())

    filename = f"{model_id.replace('/', '_')}.ipynb"
    return StreamingResponse(
        buffer,
        media_type="application/x-ipynb+json",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def classifica_modelo_sync(model_id: str, token: str, selected_weight_file:str = None) -> Dict[str, Any]:
    """Versión sincrónica usando el nuevo model_helper."""
    try:
        info = api.model_info(model_id, token=token, files_metadata=True)
    except HfHubHTTPError as err:
        raise HTTPException(404, f"Modelo '{model_id}' no encontrado: {err}")

    files = [s.rfilename for s in info.siblings]
    size_bytes = sum(s.size or 0 for s in info.siblings)

    model_info = {
        "model_id": model_id,
        "task": info.pipeline_tag or "unknown",
        "tags": info.tags or [],
        "library": info.library_name,
        "total_weight_size": size_bytes,
        "all_files": files,
        "selected_weight_file": selected_weight_file,
    }

    enriched = model_helper.process_model_info(model_info)

    # Elegir archivo de pesos por defecto
    if enriched["available_weight_files"]:
        enriched["weight_file"] = enriched["available_weight_files"][0]

    return enriched

@router.get("/get_model_extension_files", response_model=list[str])
def get_model_extension_files():
    # Devolver lista sin el punto (por ejemplo: "gguf" en lugar de ".gguf")
    extensions = sorted(ext.lstrip(".") for ext in EXT_MODELO)
    return JSONResponse(content=jsonable_encoder(extensions))


CATEGORIA_POR_PIPELINE = {
    "text-classification": "nlp",
    "text-generation": "nlp",
    "text2text-generation": "nlp",
    "token-classification": "nlp",
    "translation": "nlp",
    "summarization": "nlp",
    "question-answering": "nlp",

    "image-classification": "cv",
    "image-to-text": "cv",
    "object-detection": "cv",
    "image-segmentation": "cv",

    "audio-classification": "audio",
    "automatic-speech-recognition": "audio",
    "text-to-speech": "audio",

    "text-to-image": "multimodal",
    "image-to-image": "multimodal",
    "visual-question-answering": "multimodal",
    "zero-shot-image-classification": "multimodal",
}

@router.get("/buscar_modelos")
def buscar_modelos_paginado(
    nombre: Optional[str] = Query(None),
    categoria: Optional[str] = Query(None),
    pipeline: Optional[str] = Query(None),
    tags: Optional[List[str]] = Query(None),
    limit: int = Query(21, le=100),
    offset: int = Query(0),
    vistos: Optional[List[str]] = Query(None),
    current_user: dict = Depends(get_current_user),
):
    """Versión paginada de búsqueda de modelos, compatible con scroll infinito."""
    try:
        modelos = api.list_models(limit=1000, full=True, token=current_user.hf_token)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error accediendo a Hugging Face: {str(e)}")

    resultados = []
    vistos_set = set(vistos or [])

    for model in modelos:
        pipeline_tag = getattr(model, "pipeline_tag", None)
        model_categoria = CATEGORIA_POR_PIPELINE.get(pipeline_tag, "unknown")

        # Filtros
        if nombre and nombre.lower() not in model.modelId.lower():
            continue
        if categoria and model_categoria != categoria.lower():
            continue
        if pipeline and pipeline_tag != pipeline:
            continue
        if tags and not all(tag in model.tags for tag in tags):
            continue
        if model.modelId in vistos_set:
            continue

        resultados.append({
            "model_id": model.modelId,
            "pipeline": pipeline_tag,
            "categoria": model_categoria,
            "tags": model.tags,
            "sha": model.sha,
            "private": model.private,
        })

    total = len(resultados)
    paginados = resultados[offset:offset + limit]
    next_offset = offset + limit

    return JSONResponse(content={
        "results": paginados,
        "has_more": next_offset < total,
        "next_offset": next_offset,
    })



@router.get("/categorias_disponibles")
def categorias_disponibles():
    """
    Devuelve las categorías generales (nlp, cv, audio, multimodal) y los pipelines específicos asociados.
    """
    categorias: Dict[str, List[str]] = {}
    for pipeline, categoria in CATEGORIA_POR_PIPELINE.items():
        categorias.setdefault(categoria, []).append(pipeline)

    return JSONResponse(content=categorias)