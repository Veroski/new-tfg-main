from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Dict, Any

# Constantes importadas del archivo original
CTRANS_SUPPORTED = {
    "llama", "gptj", "gpt_neox", "opt", "mpt", "falcon", "replit",
    "pythia", "baichuan", "starcoder", "gpt_bigcode", "gemma",
    "qwen", "internlm", "rwkv", "chatglm"
}

DIFFUSERS_TASKS = {"text-to-image", "image-to-image", "image-to-text", "text-to-video", "inpainting"}

@dataclass
class Rule:
    """Regla para la selección de backend basada en predicados."""
    name: str
    pred: Callable[[dict], bool]
    backend: str
    score: int = 100  # mayor = mayor prioridad

class BackendSelector:
    """
    Selector de backend basado en reglas.
    
    Utiliza un sistema de reglas ordenadas para determinar el backend más adecuado
    para un modelo específico basado en sus características.
    """
    def __init__(self, info: dict):
        self.info = info
        self.rules: List[Rule] = self._default_rules()

    def recommend(self) -> str:
        """
        Recomienda el backend más adecuado basado en las reglas definidas.
        
        Returns:
            str: Nombre del backend recomendado
        """
        candidates = [r for r in self.rules if r.pred(self.info)]
        if not candidates:
            return "transformers"  # fallback
        # elige la de mayor puntuación (o la primera si empatan)
        return sorted(candidates, key=lambda r: r.score, reverse=True)[0].backend

    def _default_rules(self) -> List[Rule]:
        """
        Define las reglas por defecto para la selección de backend.
        
        Returns:
            List[Rule]: Lista de reglas ordenadas por prioridad
        """
        return [
            # GGUF / GGML
            Rule("gguf-ctrans", 
                 lambda i: i.get("weight_format") == "gguf" and 
                           (i.get("model_type", "").lower() in CTRANS_SUPPORTED),
                 backend="ctransformers", 
                 score=200),

            Rule("gguf-llama-cpp",  
                 lambda i: i.get("weight_format") == "gguf",
                 backend="llama-cpp-python"),

            # GPTQ
            Rule("gptq",           
                 lambda i: i.get("weight_format", "").startswith("gptq"),
                 backend="auto-gptq"),

            # AWQ
            Rule("awq",            
                 lambda i: any(f.endswith(".awq") for f in i.get("available_weight_files", [])) or
                           "awq" in i.get("weight_format", "").lower(),
                 backend="autoawq"),

            # ExLlama
            Rule("exllama",
                 lambda i: (any("exllama" in f.lower() for f in i.get("available_weight_files", [])) or
                           "exllama" in i.get("model_id", "").lower()) and
                           i.get("weight_format") == "safetensors",
                 backend="exllama"),

            # Marlin
            Rule("marlin",
                 lambda i: any(f.endswith(".marlin") for f in i.get("available_weight_files", [])) or
                           "marlin" in i.get("model_id", "").lower(),
                 backend="marlin"),

            # MLC-LLM
            Rule("mlc-llm",
                 lambda i: any(f.endswith(".mlc") for f in i.get("available_weight_files", [])) or
                           any("mlc-chat-config.json" in f.lower() for f in i.get("available_files", [])),
                 backend="mlc_llm"),

            # ONNX
            Rule("onnx",           
                 lambda i: i.get("weight_format") == "onnx",
                 backend="onnxruntime"),

            # vLLM para modelos grandes no cuantizados
            Rule("vllm",
                 lambda i: (i.get("modality") == "text" and 
                           i.get("task") == "text-generation" and
                           i.get("param_count_estimate", 0) > 10_000_000_000 and
                           i.get("quant", "none") == "none"),
                 backend="vllm"),

            # Diffusers
            Rule("diffusers",      
                 lambda i: i.get("library") == "diffusers" or i.get("task") in DIFFUSERS_TASKS,
                 backend="diffusers"),

            # Transformers 8-bit para modelos grandes
            Rule("transformers-8bit",
                 lambda i: i.get("modality") == "text" and 
                           i.get("task") == "text-generation" and
                           i.get("param_count_estimate", 0) > 6_000_000_000,
                 backend="transformers-8bit"),

            # Fallback genérico
            Rule("transformers",   
                 lambda i: True,   # catch-all
                 backend="transformers", 
                 score=0),
        ]
