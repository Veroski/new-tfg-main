import os
import re
from typing import Tuple

# ---------------- Tags válidos -----------------
_TAG_REGEX = re.compile(r"^[A-Za-z0-9_-]+$")

def es_tag_valido(tag: str) -> bool:
    return ":" not in tag and _TAG_REGEX.fullmatch(tag) is not None


# ---------------- Extensiones “modelo” ----------
EXT_MODELO = {
    ".gguf", ".ggml",
    ".safetensors", ".bin", ".pt", ".pth",
    ".ckpt", ".onnx", ".tflite",
    ".h5", ".pb", ".msgpack",
    ".ot", ".mlmodel",
    ".pkl", ".joblib",
}

def es_archivo_modelo(nombre: str) -> bool:
    return os.path.splitext(nombre)[1].lower() in EXT_MODELO


# ---------------- Calidad / variante ------------
_PATTERNS = [
    (re.compile(r"(?:f|fp)32", re.I), "fp32", 5, "Máxima precisión; VRAM alta"),
    (re.compile(r"(?:f|bf)16", re.I), "fp16/bf16", 4,
     "Muy precisa; buen equilibrio en GPU modernas"),
    (re.compile(r"q8", re.I), "q8", 3,
     "8 bits: buena calidad con ahorro de memoria"),
    (re.compile(r"q6|int6", re.I), "q6/int6", 3,
     "6 bits: calidad razonable para CPU"),
    (re.compile(r"q5", re.I), "q5", 2,
     "5 bits: ligera pérdida, menos RAM"),
    (re.compile(r"q4|int4", re.I), "q4/int4", 2,
     "4 bits: mínima RAM, pérdida perceptible"),
    (re.compile(r"q[23]|int[23]", re.I), "≤4 bits", 1,
     "Compresión extrema; puede degradar la salida"),
]

def analizar_calidad(nombre: str, size_bytes: int | None) -> Tuple[str, int, str]:
    """Devuelve (variante, rank 1–5, nota)"""
    for rgx, var_, rank_, nota_ in _PATTERNS:
        if rgx.search(nombre):
            return var_, rank_, nota_

    # Heurística por extensión/tamaño cuando no hay patrón
    ext = os.path.splitext(nombre)[1].lower()
    if ext in {".safetensors", ".bin", ".pt", ".pth"}:
        if size_bytes and size_bytes > 8 * 1024**3:
            return "fp32≈", 5, "Tamaño muy grande: probablemente fp32"
        if size_bytes and size_bytes > 2 * 1024**3:
            return "fp16≈", 4, "Probable fp16 (medio‑alto)"
        return "int8≈", 3, "Tamaño moderado: posible 8 bits"

    return "?", 0, "Sin indicios de precisión"


# ---------------- Compatibilidad Colab ----------
def evaluar_colab(size_gb: float | None, rank: int) -> tuple[str, str]:
    if size_gb is None:
        return "?", "Tamaño desconocido"
    if size_gb > 15:
        return "❌", f"{size_gb} GB: Demasiado grande para Colab (VRAM 15 GB)"
    if size_gb > 8:
        msg = (f"{size_gb} GB: Solo Colab Pro/T4 (o superior)")
        if rank >= 4:
            msg += " — precisión alta"
        return "⚠️", msg
    return "✅", f"{size_gb} GB: Ok en Colab free/pro"


# ---------------- Agrupado de shards ------------
SHARD_RGX = re.compile(r"^(.*?)[_-]\d{5}-of-\d{5}\.safetensors$", re.I)


def nombre_grupo_shards(prefijo: str, n_shards: int) -> str:
    return f"{prefijo}.safetensors ({n_shards} shards)"


# ----------------------------------------
# NUEVO: /filtrar_modelos (con pipeline_tag)
# ----------------------------------------
PIPELINES: dict[str, list[str]] = {
    "audio": [
        "text-to-speech", "text-to-audio", "automatic-speech-recognition",
        "audio-to-audio", "audio-classification", "voice-activity-detection",
        "audio-text-to-text",
    ],
    "imagen": [
        "image-text-to-text", "visual-question-answering", "document-question-answering",
        "depth-estimation", "image-classification", "object-detection",
        "image-segmentation", "text-to-image", "image-to-text", "image-to-image",
        "image-to-video", "unconditional-image-generation",
        "zero-shot-image-classification", "mask-generation",
        "zero-shot-object-detection", "text-to-3d", "image-to-3d",
        "image-feature-extraction", "keypoint-detection",
    ],
    "video": [
        "video-text-to-text", "visual-document-retrieval", "video-classification",
        "text-to-video",
    ],
    "texto": [
        "text-classification", "token-classification", "table-question-answering",
        "question-answering", "zero-shot-classification", "translation",
        "summarization", "feature-extraction", "text-generation",
        "text2text-generation", "fill-mask", "sentence-similarity",
        "text-ranking",
    ],

    "general": [],
}



