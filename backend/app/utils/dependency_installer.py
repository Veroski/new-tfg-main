from __future__ import annotations
from typing import List, Dict, Callable, Any

class DependencyInstaller:
    """
    Gestor de instalación de dependencias para diferentes backends.
    
    Utiliza un sistema de handlers registrados para determinar los comandos
    de instalación necesarios para cada backend.
    """
    
    def __init__(self):
        self.install_handlers = self._default_handlers()
    
    def get_install_commands(self, backend: str, modality: str = "text") -> List[str]:
        """
        Obtiene los comandos de instalación para un backend específico.
        
        Args:
            backend: Nombre del backend
            modality: Modalidad del modelo (text, vision, audio)
            
        Returns:
            List[str]: Lista de comandos de instalación
        """
        # Caso especial para transformers
        if backend.startswith("transformers"):
            extra = "[vision]" if modality == "vision" else "[audio]" if modality == "audio" else ""
            base = [f"pip install -q --upgrade transformers{extra}"]
            if backend.endswith("8bit"):
                base.append("pip install -q bitsandbytes")
            return base
        
        # Usar el handler registrado o devolver lista vacía si no existe
        handler = self.install_handlers.get(backend)
        if handler:
            return handler(modality)
        return []
    
    def _default_handlers(self) -> Dict[str, Callable[[str], List[str]]]:
        """
        Define los handlers por defecto para la instalación de dependencias.
        
        Returns:
            Dict[str, Callable]: Diccionario de handlers por backend
        """
        return {
            "ctransformers": lambda _: ["pip install -q ctransformers"],
            
            "llama-cpp-python": lambda _: ["pip install -q llama-cpp-python huggingface-hub"],
            
            "auto-gptq": lambda _: [
                "pip install -q auto-gptq", 
                "pip install -q --upgrade transformers"
            ],
            
            "autoawq": lambda _: ["pip install -q autoawq accelerate"],
            
            "exllama": lambda _: ["pip install -q exllama"],
            
            "marlin": lambda _: ["pip install -q marlin"],
            
            "mlc_llm": lambda _: ["pip install -q mlc-llm"],
            
            "vllm": lambda _: ["pip install -q vllm"],
            
            "onnxruntime": lambda _: [
                "pip install -q optimum onnxruntime-gpu || pip install -q optimum onnxruntime"
            ],
            
            "diffusers": lambda _: ["pip install -q diffusers accelerate transformers"],
            
            "speechbrain": lambda _: ["pip install -q speechbrain"],
            
            "espnet": lambda _: ["pip install -q espnet"],
        }
    
    def register_handler(self, backend: str, handler: Callable[[str], List[str]]) -> None:
        """
        Registra un nuevo handler para un backend específico.
        
        Args:
            backend: Nombre del backend
            handler: Función que genera los comandos de instalación
        """
        self.install_handlers[backend] = handler
