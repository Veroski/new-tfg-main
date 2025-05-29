from __future__ import annotations
from typing import Dict, Any
import nbformat as nbf
from .model_helper_updated import process_model_info
from .notebook_builder import create_notebook_builder

def create_notebook(model_id: str, model_info: Dict[str, Any] = None) -> nbf.NotebookNode:
    """
    Crea un notebook para un modelo específico.
    
    Args:
        model_id: ID del modelo en Hugging Face
        model_info: Información del modelo (opcional, si ya se ha procesado)
        
    Returns:
        nbf.NotebookNode: Notebook generado
    """
    # Si no se proporciona información del modelo, crear un diccionario básico
    if model_info is None:
        model_info = {
            "model_id": model_id,
            "task": "text-generation",  # valor por defecto
            "tags": [],
            "library": "",
            "total_weight_size": 0,
            "all_files": [],
        }
    
    # Asegurarse de que model_id esté en la información
    if "model_id" not in model_info:
        model_info["model_id"] = model_id
    
    # Procesar la información del modelo si no tiene backend recomendado
    if "recommended_backend" not in model_info:
        model_info = process_model_info(model_info)
    
    # Crear el constructor de notebooks adecuado
    builder = create_notebook_builder(model_info)
    
    # Construir y devolver el notebook
    return builder.build()