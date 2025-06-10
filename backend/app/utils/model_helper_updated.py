from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Dict, List
import nbformat as nbf
import os
from pathlib import Path
from textwrap import dedent

# Importamos los nuevos módulos
from .backend_selector import BackendSelector
from .format_detector import detect_format, is_quantized, get_weight_files
from .dependency_installer import DependencyInstaller
from .notebook_builder import create_notebook_builder

# Constantes originales mantenidas para compatibilidad
TEXT_TASKS = {
    "text-generation",
    "text-classification",
    "summarization",
    "translation",
    "question-answering",
    "conversational",
    "fill-mask",
}
VISION_TASKS = {
    "image-classification",
    "object-detection",
    "image-segmentation",
    "image-to-text",
}
AUDIO_TASKS = {
    "automatic-speech-recognition",
    "audio-classification",
}

DIFFUSERS_TASKS = {"text-to-image", "image-to-image", "image-to-text", "text-to-video", "inpainting"}

CTRANS_SUPPORTED = {
    "llama", "gptj", "gpt_neox", "opt", "mpt", "falcon", "replit",
    "pythia", "baichuan", "starcoder", "gpt_bigcode", "gemma",
    "qwen", "internlm", "rwkv", "chatglm"
}

# ---------------
# Helper methods
# ---------------

def _estimate_param_count(size_bytes: int, quant: str) -> int:
    """Very rough param count estimate from total checkpoint size."""
    if size_bytes == 0:
        return 0
    # assume fp32 (4 bytes) unless quantized. fp16 ~2 bytes, 8bit ~1 byte, 4bit ~0.5 byte
    bytes_per_param = 4.0  # fp32 default
    if quant == "none":
        # try to infer if fp16 by size heuristic (<2GB for 7B params) but keep simple
        pass
    elif quant in ["gptq", "awq", "gguf-4bit", "marlin-4bit", "exllama"]:
        bytes_per_param = 0.5  # 4‑bit
    # convert
    params = int(size_bytes / bytes_per_param)
    return params


def _infer_modality(task: str, tags: List[str]) -> str:
    if task in TEXT_TASKS or any(t in TEXT_TASKS for t in tags):
        return "text"
    if task in VISION_TASKS or any(t in VISION_TASKS for t in tags):
        return "vision"
    if task in AUDIO_TASKS or any(t in AUDIO_TASKS for t in tags):
        return "audio"
    # fallback
    return "multimodal"


def _task_group(task: str) -> str:
    if task in TEXT_TASKS:
        return "text"
    if task in VISION_TASKS:
        return "vision"
    if task in AUDIO_TASKS:
        return "audio"
    return "other"


# -------------------------------------------------------------------
# Función principal para procesar la información del modelo
# -------------------------------------------------------------------
def process_model_info(model_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Procesa la información del modelo y determina el backend más adecuado,
    usando un archivo de pesos específico si se ha indicado.
    
    Args:
        model_info: Diccionario con información del modelo
        
    Returns:
        Dict[str, Any]: Información del modelo enriquecida con backend recomendado
    """
    # Extraer información relevante
    files = model_info.get("all_files", [])
    selected_file = model_info.get("selected_weight_file")  # <- NUEVO

    if selected_file and selected_file in files:
        weight_files = [selected_file]
    else:
        weight_files = get_weight_files(files)
        selected_file = weight_files[0] if weight_files else None
    
    # Detectar formato y cuantización
    weight_format = detect_format([selected_file]) if selected_file else "unknown"
    quant = is_quantized(weight_format)
    
    # Inferir modalidad
    task = model_info.get("task", "")
    tags = model_info.get("tags", [])
    modality = _infer_modality(task, tags)
    
    # Estimar parámetros
    size_bytes = model_info.get("total_weight_size", 0)
    param_count = _estimate_param_count(size_bytes, quant)
    
    # Enriquecer la información del modelo
    enriched_info = {
        **model_info,
        "weight_file": selected_file,  # <- usado por el notebook
        "weight_format": weight_format,
        "quant": quant,
        "modality": modality,
        "param_count_estimate": param_count,
        "available_weight_files": weight_files,
    }
    
    # Seleccionar backend
    selector = BackendSelector(enriched_info)
    recommended_backend = selector.recommend()
    
    # Añadir backend recomendado
    enriched_info["recommended_backend"] = recommended_backend
    
    return enriched_info


# -------------------------------------------------------------------
# Función para generar el notebook
# -------------------------------------------------------------------
def build_notebook(model_info: Dict[str, Any]) -> nbf.NotebookNode:
    """
    Construye un notebook para el modelo especificado.
    
    Args:
        model_info: Información del modelo enriquecida
        
    Returns:
        nbf.NotebookNode: Notebook generado
    """
    # Procesar información si es necesario
    if "recommended_backend" not in model_info:
        model_info = process_model_info(model_info)
    
    # Crear el constructor de notebooks adecuado
    builder = create_notebook_builder(model_info)
    
    # Construir y devolver el notebook
    return builder.build()


# -------------------------------------------------------------------
# Función para obtener comandos de instalación
# -------------------------------------------------------------------
def get_install_commands(backend: str, modality: str = "text") -> List[str]:
    """
    Obtiene los comandos de instalación para un backend específico.

    Args:
        backend: Nombre del backend
        modality: Modalidad del modelo
        
    Returns:
        List[str]: Lista de comandos de instalación
    """
    installer = DependencyInstaller()
    return installer.get_install_commands(backend, modality)
