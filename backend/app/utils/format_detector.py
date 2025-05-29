from __future__ import annotations
from pathlib import Path
import re
from typing import List

# Patrones y extensiones
WEIGHT_EXTS = (".bin", ".safetensors", ".pt", ".ggml", ".gguf", ".onnx", ".awq", ".marlin", ".mlc")
GGUF_EXTS = (".gguf", ".ggml")
GPTQ_PAT = re.compile(r"gptq", re.I)
EXLLAMA_PAT = re.compile(r"exllama", re.I)

# Mapa de extensiones adicionales a formatos
_EXTRA_EXTS = {
    ".awq": "awq",
    ".marlin": "marlin",
    ".mlc": "mlc",
}

def get_weight_files(files: List[str]) -> List[str]:
    """Return list of files that look like weight checkpoints."""
    return [f for f in files if Path(f).suffix.lower() in WEIGHT_EXTS]

def detect_format(weight_files: List[str]) -> str:
    """
    Identify the primary weight format based on file extensions and patterns.
    
    Args:
        weight_files: Lista de archivos de pesos del modelo
        
    Returns:
        str: Formato detectado (gguf, onnx, awq, marlin, mlc, etc.)
    """
    # Primero verificamos extensiones especiales
    for f in weight_files:
        ext = Path(f).suffix.lower()
        if ext in _EXTRA_EXTS:
            return _EXTRA_EXTS[ext]
    
    # Verificamos archivos de configuración específicos
    all_files_lower = [f.lower() for f in weight_files]
    if any("awq_config.json" in f for f in all_files_lower):
        return "awq"
    if any("mlc-chat-config.json" in f for f in all_files_lower):
        return "mlc"
    
    # Verificamos patrones en nombres de archivos
    if any(EXLLAMA_PAT.search(f) for f in weight_files) and any(f.lower().endswith(".safetensors") for f in weight_files):
        return "exllama-safetensors"
    
    # Verificamos extensiones estándar
    if any(f.lower().endswith(GGUF_EXTS) for f in weight_files):
        return "gguf"
    if any(f.lower().endswith(".onnx") for f in weight_files):
        return "onnx"
    if any(f.lower().endswith(".safetensors") for f in weight_files):
        if any(GPTQ_PAT.search(f) for f in weight_files):
            return "gptq-safetensors"
        return "safetensors"
    if any(f.lower().endswith(".pt") for f in weight_files):
        return "torchscript"
    
    # default
    return "pytorch-bin"

def is_quantized(weight_format: str) -> str:
    """
    Determina el tipo de cuantización basado en el formato de pesos.
    
    Args:
        weight_format: Formato de pesos detectado
        
    Returns:
        str: Tipo de cuantización (gptq, awq, gguf-4bit, etc.) o "none"
    """
    if "gptq" in weight_format:
        return "gptq"
    if weight_format == "gguf":
        return "gguf-4bit"
    if weight_format == "awq":
        return "awq"
    if "exllama" in weight_format:
        return "exllama"
    if weight_format == "marlin":
        return "marlin-4bit"
    return "none"
