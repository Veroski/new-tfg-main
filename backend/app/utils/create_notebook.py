from __future__ import annotations
from typing import Dict, Any
import nbformat as nbf
from .model_helper_updated import process_model_info
from .notebook_builder import create_notebook_builder

def create_notebook(model_id: str, model_info: Dict[str, Any] = None, user: dict = None) -> nbf.NotebookNode:
    """
    Crea un notebook para un modelo específico.

    Args:
        model_id: ID del modelo en Hugging Face
        model_info: Información del modelo (opcional)
        user: Usuario actual (opcional)

    Returns:
        nbf.NotebookNode: Notebook generado
    """
    if model_info is None:
        model_info = {
            "model_id": model_id,
            "task": "text-generation",
            "tags": [],
            "library": "",
            "total_weight_size": 0,
            "all_files": [],
        }

    if "model_id" not in model_info:
        model_info["model_id"] = model_id

    if "recommended_backend" not in model_info:
        model_info = process_model_info(model_info)

    builder = create_notebook_builder(model_info, user=user)
    return builder.build()
