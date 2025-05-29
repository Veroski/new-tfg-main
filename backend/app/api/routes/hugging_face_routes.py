import json
import anyio
from fastapi import APIRouter, HTTPException, Query
from typing import Any, List, Optional, Dict
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
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
from app.api.helpers import (
    es_archivo_modelo,
    es_tag_valido,
    analizar_calidad,
    evaluar_colab,
    SHARD_RGX,
    nombre_grupo_shards,
    PIPELINES
)

router = APIRouter()
api    = HfApi()
token  = settings.token


# ---------- /buscar_modelos ---------------------------------
@router.get("/buscar_modelos")
def buscar_modelos_llm(
    tarea: str = Query("text-generation"),
    incluir_palabras: Optional[List[str]] = Query(None),
    incluir_tags: Optional[List[str]] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
):
    # Pedimos una cantidad suficientemente grande para filtrar luego
    raw = api.list_models(
        filter=tarea,
        sort="downloads",
        direction=-1,
        limit=page * page_size * 2,  # Pedimos más de lo necesario por seguridad
        token=token,
    )

    resultados = []
    for modelo in raw:
        nombre = modelo.modelId.lower()

        if incluir_palabras and not any(p in nombre for p in map(str.lower, incluir_palabras)):
            continue

        tags_limpios = [t for t in (modelo.tags or []) if es_tag_valido(t)]
        if incluir_tags and not any(t in tags_limpios for t in incluir_tags):
            continue

        resultados.append(
            {
                "nombre": modelo.modelId,
                "descargas": modelo.downloads,
                "ultima_modificacion": modelo.lastModified,
                "tags": tags_limpios,
            }
        )

    # Paginado manual sobre los resultados filtrados
    start = (page - 1) * page_size
    end = start + page_size
    pagina = resultados[start:end]

    return JSONResponse(
        content=jsonable_encoder({
            "page": page,
            "page_size": page_size,
            "total_returned": len(pagina),
            "results": pagina,
            "has_more": len(resultados) > end,
        })
    )


# ---------- /detalles_modelo/{id} ----------------------------
@router.get("/detalles_modelo/{model_id:path}")
def obtener_tamanos_archivos(model_id: str):
    info = api.model_info(repo_id=model_id, token=token, files_metadata=True)

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


@router.get("/buscar_modelos_live")
def buscar_modelos_live(
    q: str                                = Query(..., min_length=2, description="Texto que escribe el usuario"),
    tarea: str                            = Query("text-generation"),
    incluir_tags: Optional[List[str]]     = Query(None),
    page: int                             = Query(1, ge=1),
    page_size: int                        = Query(10, ge=1, le=100),
):
    """
    Devuelve modelos que coinciden con `q` usando el buscador de la HuggingFace Hub,
    además de los filtros habituales: tarea, tags, paginación.
    """
    # Pedimos un "colchón" para filtrar localmente
    raw = api.list_models(
        search=q,                          # ← ¡Búsqueda real en HF!
        filter=tarea,
        sort="downloads",
        direction=-1,
        limit=page * page_size * 2,
        token=token,
    )

    resultados = []
    for modelo in raw:
        tags_limpios = [t for t in (modelo.tags or []) if es_tag_valido(t)]
        if incluir_tags and not any(t in tags_limpios for t in incluir_tags):
            continue

        resultados.append(
            {
                "nombre": modelo.modelId,
                "descargas": modelo.downloads,
                "ultima_modificacion": modelo.lastModified,
                "tags": tags_limpios,
            }
        )

    # Paginado
    start, end = (page - 1) * page_size, page * page_size
    pagina = resultados[start:end]

    return JSONResponse(
        content=jsonable_encoder({
            "page": page,
            "page_size": page_size,
            "total_returned": len(pagina),
            "results": pagina,
            "has_more": len(resultados) > end,
        })
    )
from fastapi import Body
@router.get("/filtrar_modelos")
def filtrar_modelos(
    categoria: str = Query("general", enum=list(PIPELINES)),
    tarea: str     = Query(...),
    q: str         = Query(""),
    sort: str      = Query("downloads", pattern="^(downloads|likes|created)$"),
    page_size: int = Query(21, ge=1, le=100),
    offset: int    = Query(0,  ge=0),                     # dónde empezar a buscar
    seen:  Optional[str] = Query(None, description="CSV de IDs ya mostrados"),
):
    if categoria != "general" and tarea not in PIPELINES.get(categoria, []):
        tareas_posibles = PIPELINES[categoria]
        if not tareas_posibles:
            raise HTTPException(status_code=400, detail=f"No hay tareas válidas para '{categoria}'")
        tarea = tareas_posibles[0]

    ids_vistos = set(seen.split(",")) if seen else set()

    url          = "https://huggingface.co/api/models"
    batch_limit  = 500                                   # máximo que permite la API
    internal_off = offset                                # offset que vamos moviendo internamente
    nuevos: list[dict] = []

    while len(nuevos) < page_size:
        params = {
            "pipeline_tag": tarea,
            "sort": sort,
            "limit": batch_limit,
            "offset": internal_off,
        }
        if len(q.strip()) >= 2:
            params["search"] = q.strip()

        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            lote = r.json()
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Error consultando Hugging Face: {str(e)}")

        if not lote:                                     # sin más resultados
            break

        for m in lote:
            mid = m.get("modelId")
            if not mid or mid in ids_vistos:
                continue

            nuevos.append({
                "nombre": mid,
                "descargas": m.get("downloads"),
                "ultima_modificacion": m.get("lastModified"),
                "tags": [t for t in (m.get("tags") or []) if es_tag_valido(t)],
            })
            ids_vistos.add(mid)

            if len(nuevos) == page_size:
                break

        internal_off += batch_limit                      # avanzamos para siguiente ronda
        if len(lote) < batch_limit:                      # Hugging Face se quedó sin más
            break

    next_offset = internal_off if len(nuevos) == page_size else None

    return JSONResponse(content=jsonable_encoder({
        "results": nuevos,
        "page_size": page_size,
        "has_more": next_offset is not None,
        "next_offset": next_offset,
    }))


# ---- /route_model/{id} ---------------------------------
@router.get("/route_model/{model_id:path}")
async def route_model(model_id: str):
    """
    Ruta principal para generar un notebook para un modelo.
    Devuelve un JSON con la URL del notebook generado.
    """
    try:
        # Obtener información del modelo
        info = await anyio.to_thread.run_sync(classifica_modelo_sync, model_id)
        
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


# ---- /classifica_modelo ---------------------------------
@router.get("/classifica_modelo/{model_id:path}")
async def classifica_modelo(model_id: str):
    """Lanza la recogida de metadatos en un hilo y devuelve JSON."""
    try:
        info = await anyio.to_thread.run_sync(classifica_modelo_sync, model_id)
    except HTTPException as e:
        # re-levantar para que FastAPI devuelva el código correcto
        raise e
    return JSONResponse(content=info)

# ---- /genera_notebook -----------------------------------
@router.get("/genera_notebook/{model_id:path}")
async def genera_notebook(model_id: str):
    """Genera el .ipynb en segundo plano y lo devuelve como descarga."""
    info = await anyio.to_thread.run_sync(classifica_modelo_sync, model_id)
    
    # Usar la función create_notebook en lugar de model_helper.build_notebook
    nb = await anyio.to_thread.run_sync(create_notebook, model_id, info)

    data = await anyio.to_thread.run_sync(nbf.writes, nb)
    buffer = BytesIO(data.encode())

    filename = f"{model_id.replace('/', '_')}.ipynb"
    return StreamingResponse(
        buffer,
        media_type="application/x-ipynb+json",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def classifica_modelo_sync(model_id: str) -> Dict[str, Any]:
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
    }

    enriched = model_helper.process_model_info(model_info)

    # Elegir archivo de pesos por defecto
    if enriched["available_weight_files"]:
        enriched["weight_file"] = enriched["available_weight_files"][0]

    return enriched
