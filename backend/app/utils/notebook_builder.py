


from __future__ import annotations
from datetime import datetime
from textwrap import dedent
from typing import Dict, Any, List
import nbformat as nbf

class NotebookBuilder:
    """
    Constructor modular de notebooks para modelos de Hugging Face.
    
    Permite construir notebooks de forma modular, separando cada sección
    en métodos independientes que pueden ser sobrescritos en subclases.
    """
    
    def __init__(self, info: Dict[str, Any], user: dict = None):
        """
        Inicializa el constructor de notebooks.
        
        Args:
            info: Diccionario con información del modelo
        """
        self.nb = nbf.v4.new_notebook()
        self.info = info
        self.user = user
    
    def build(self) -> nbf.NotebookNode:
        """
        Construye el notebook completo.
        
        Returns:
            nbf.NotebookNode: Notebook generado
        """

        self.add_metadata()
        self.add_hftoken()
        self.add_install()
        self.add_gpu_check()
        self.add_download_weights()
        self.add_readme_preview()
        
        # Eliminar la verificación de generate() con exec
        # model_setup_code = self._model_setup_snippet(self.info)
        # try:
        #     tmp_globals = {}
        #     exec(model_setup_code, tmp_globals)
        #     assert 'generate' in tmp_globals and callable(tmp_globals['generate']), \
        #         "El snippet debe definir una función generate() callable"
        # except Exception as e:
        #     print(f"Warning: Snippet verification failed: {e}")
        
        self.add_model_setup()
        self.add_prompt_cell()
        self.add_inference_cell()
        self.add_result_processing_cell()
        self.add_footer()
        return self.nb
    
    def add_metadata(self) -> None:
        """Añade la celda de metadatos y título al notebook."""
        md_table = "\n".join(
            [
                "| Campo | Valor |",
                "|-------|-------|",
                f"| **Tarea / Task** | {self.info.get('task', 'N/A')} |",
                f"| **Modalidad** | {self.info.get('modality', 'N/A')} |",
                f"| **Backend sugerido** | {self.info.get('recommended_backend', 'N/A')} |",
                f"| **Archivo de pesos** | {self.info.get('weight_file', 'N/A')} |",
                f"| **Formato** | {self.info.get('weight_format', 'N/A')} |",
                f"| **Parámetros (est.)** | {self.info.get('param_count_estimate', 0):,} |",
            ]
        )

        self.nb.cells.append(
            nbf.v4.new_markdown_cell(
                f"""# 📓 Notebook para `{self.info.get('model_id', 'unknown')}`

*Autogenerado: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*

{md_table}
"""
            )
        )
    
    def add_install(self) -> None:
        """Añade la celda de instalación de dependencias."""
        from app.utils.dependency_installer import DependencyInstaller
        
        installer = DependencyInstaller()
        install_cmds = installer.get_install_commands(
            self.info.get('recommended_backend', 'transformers'), 
            self.info.get('modality', 'text')
        )
        
        self.nb.cells.append(
            nbf.v4.new_code_cell(self._install_cell(install_cmds))
        )
    
    def add_hftoken(self) -> None:
        """Añade la celda para configurar el token de Hugging Face."""
        print("USER: ", self.user)
        hf_token = self.user.hf_token
        if hf_token:
            self.nb.cells.append(
                nbf.v4.new_code_cell(
                    dedent(
                        f"""
                        import os
                        os.environ['HF_TOKEN'] = '{hf_token}'
                        print('✅ Token de Hugging Face configurado')
                        """
                    )
                )
            )
        
    def add_gpu_check(self) -> None:
        """Añade la celda de verificación de GPU."""
        self.nb.cells.append(
            nbf.v4.new_code_cell(self._gpu_self_test_cell())
        )
    
    def add_download_weights(self) -> None:
        """Añade la celda de descarga de pesos si es necesario."""
        weight_format = self.info.get('weight_format', '')
        if weight_format in {"gguf", "ggml", "awq", "marlin", "mlc"}:
            self.nb.cells.append(
                nbf.v4.new_code_cell(
                    dedent(
                        f"""
                        from huggingface_hub import hf_hub_download
                        WEIGHT_FILE = hf_hub_download('{self.info.get('model_id', '')}', filename='{self.info.get('weight_file', '')}', local_dir='.', local_dir_use_symlinks=False)
                        print('✅ Pesos descargados en', WEIGHT_FILE)
                        """
                    )
                )
            )
    
    def add_readme_preview(self) -> None:
        """Añade la celda de previsualización del README."""
        self.nb.cells.append(
            nbf.v4.new_code_cell(self._readme_cell(self.info.get('model_id', '')))
        )
    
    def add_model_setup(self) -> None:
        """Añade la celda de configuración del modelo."""
        self.nb.cells.append(
            nbf.v4.new_code_cell(self._model_setup_snippet(self.info))
        )
    
    def add_prompt_cell(self) -> None:
        """Añade la celda para definir el prompt."""
        task = self.info.get('task', '')
        modality = self.info.get('modality', '')
        
        prompt_cell = self._get_prompt_cell(task, modality)
        self.nb.cells.append(nbf.v4.new_code_cell(prompt_cell))
    
    def add_inference_cell(self) -> None:
        """Añade la celda para ejecutar la inferencia con el prompt."""
        task = self.info.get('task', '')
        modality = self.info.get('modality', '')
        
        inference_cell = self._get_inference_cell(task, modality)
        self.nb.cells.append(nbf.v4.new_code_cell(inference_cell))
    
    def add_result_processing_cell(self) -> None:
        """Añade la celda para procesar y guardar el resultado si es necesario."""
        task = self.info.get('task', '')
        modality = self.info.get('modality', '')
        
        # Solo añadir esta celda para tareas que generan resultados que se pueden guardar
        if self._needs_result_processing(task, modality):
            processing_cell = self._get_result_processing_cell(task, modality)
            self.nb.cells.append(nbf.v4.new_code_cell(processing_cell))
    
    def add_footer(self) -> None:
        """Añade la celda de pie de página."""
        self.nb.cells.append(
            nbf.v4.new_markdown_cell(
                "---\n✅ *Notebook generado automáticamente. ¡Disfruta!*"
            )
        )
    
    def _install_cell(self, cmds: List[str]) -> str:
        """Genera el contenido de la celda de instalación."""
        if not cmds:
            return "print('✅ No additional packages required')"

        joined = "\n".join(f"sh('{c}')" for c in cmds)
        return dedent(
            f"""
            import subprocess, sys, torch, platform
            from datetime import datetime

            def sh(cmd: str):
                print(f"🚀 {{cmd}} ({{datetime.now().strftime('%H:%M:%S')}})")
                subprocess.run(cmd, shell=True, check=True)

            print('🐍 Python', platform.python_version())
            print('🖥️ CUDA available →', torch.cuda.is_available())
            {joined}
            """
        )
    
    def _gpu_self_test_cell(self) -> str:
        """Genera el contenido de la celda de verificación de GPU."""
        return dedent(
            """
            import torch, psutil, humanize, platform

            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                name = torch.cuda.get_device_name(idx)
                total = humanize.naturalsize(torch.cuda.get_device_properties(idx).total_memory)
                print(f"✔️ CUDA GPU detected: {name} ({total})")
            else:
                print("⚠️ CUDA not available – falling back to CPU. Expect slower inference.")
            """
        )
    
    def _readme_cell(self, model_id: str) -> str:
        """Genera el contenido de la celda de previsualización del README."""
        return dedent(
            f"""
            from huggingface_hub import hf_hub_download
            import os, pathlib, textwrap, html, io, IPython

            try:
                readme_path = hf_hub_download('{model_id}', 'README.md')
                txt = pathlib.Path(readme_path).read_text(encoding='utf-8')
                if len(txt) > 300_000:
                    print('README is too large, skipping preview.')
                else:
                    IPython.display.display(IPython.display.HTML('<details><summary><b>Model Card / Tarjeta del modelo</b></summary>'+html.escape(txt).replace('\\n','<br>')+'</details>'))
            except Exception as e:
                print('Could not fetch README:', e)
            """
        )
    
    def _model_setup_snippet(self, info: Dict[str, Any]) -> str:
        """Genera el contenido de la celda de configuración del modelo."""
        backend = info.get("recommended_backend", "")
        model_id = info.get("model_id", "")
        weight_file = info.get("weight_file", "")
        model_type = info.get("model_type", "")
        task = info.get("task", "")
        modality = info.get("modality", "")

        device_snip = "device = 'cuda' if torch.cuda.is_available() else 'cpu'"
        
        # Eliminar unified_generate_function
        # unified_generate_function = """
        # def generate(prompt: str, **gen_kwargs) -> str:
        #     if 'pipe' in globals():            # transformers pipeline
        #         out = pipe(prompt, **gen_kwargs)
        #         return out[0].get('generated_text', str(out))

        #     if 'llm' in globals():             # llama-cpp / ctransformers / exllama
        #         out = llm(prompt, **gen_kwargs)
        #         return out['choices'][0]['text'] if isinstance(out, dict) else str(out)

        #     if 'model' in globals() and 'tokenizer' in globals():  # GPTQ, AWQ, ONNX, …
        #         inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        #         ids    = model.generate(**input, **gen_kwargs)
        #         return tokenizer.decode(ids[0], skip_special_tokens=True)

        #     raise RuntimeError('No backend inference object found')

        # assert callable(generate), 'generate() debe definirse en el snippet'
        # """

        # Ejemplos para diferentes backends
        if backend == "llama-cpp-python":
            return f"""import torch
from llama_cpp import Llama

{device_snip}

# Configuración del modelo
print("Cargando modelo con llama-cpp-python...")
llm = Llama(
    model_path="{weight_file}",
    n_ctx=2048,
    n_gpu_layers=35 if torch.cuda.is_available() else 0
)
print("✅ Modelo cargado correctamente")

def generate(prompt: str, **gen_kwargs) -> str:
    out = llm(prompt, **gen_kwargs)
    return out['choices'][0]['text'] if isinstance(out, dict) else str(out)
"""

        if backend == "ctransformers":
            model_type_line = f", model_type='{model_type}'" if model_type else ""
            return f"""import torch
from ctransformers import AutoModelForCausalLM

# Configuración del modelo
print("Cargando modelo con ctransformers...")
llm = AutoModelForCausalLM.from_pretrained(
    '{model_id}',
    model_file='{weight_file}'{model_type_line},
    gpu_layers=35 if torch.cuda.is_available() else 0
)
print("✅ Modelo cargado correctamente")

def generate(prompt: str, **gen_kwargs) -> str:
    out = llm(prompt, **gen_kwargs)
    return out['choices'][0]['text'] if isinstance(out, dict) else str(out)
"""

        if backend == "auto-gptq":
            return f"""import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

{device_snip}

# Configuración del modelo
print("Cargando modelo GPTQ...")
tokenizer = AutoTokenizer.from_pretrained('{model_id}', use_fast=True)
model = AutoModelForCausalLM.from_pretrained('{model_id}', device_map='auto', trust_remote_code=True)
print("✅ Modelo cargado correctamente")

def generate(prompt: str, **gen_kwargs) -> str:
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    ids    = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(ids[0], skip_special_tokens=True)
"""

        if backend == "autoawq":
            return f"""import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

{device_snip}

# Configuración del modelo
print("Cargando modelo AWQ...")
tokenizer = AutoTokenizer.from_pretrained('{model_id}', use_fast=True)
model = AutoAWQForCausalLM.from_pretrained('{model_id}', device_map='auto')
print("✅ Modelo cargado correctamente")

def generate(prompt: str, **gen_kwargs) -> str:
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    ids    = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(ids[0], skip_special_tokens=True)
"""

        if backend == "exllama":
            return f"""import torch
from exllama.model import ExLlamaModel
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator

{device_snip}

# Configuración del modelo
print("Cargando modelo ExLlama...")
model = ExLlamaModel(model_path='{model_id}')
tokenizer = ExLlamaTokenizer(model_path='{model_id}')
generator = ExLlamaGenerator(model, tokenizer)
print("✅ Modelo cargado correctamente")

def generate(prompt: str, **gen_kwargs) -> str:
    # Caso especial para ExLlama que usa generator
    max_tokens = gen_kwargs.get('max_new_tokens', 256)
    generator.settings.token_repetition_penalty_max = gen_kwargs.get('repetition_penalty', 1.1)
    generator.settings.temperature = gen_kwargs.get('temperature', 0.7)
    generator.settings.top_p = gen_kwargs.get('top_p', 0.9)
    generator.settings.top_k = gen_kwargs.get('top_k', 40)
    
    return generator.generate(prompt, max_new_tokens=max_tokens)
"""
        
        # onnxruntime  ★ NUEVO ★
        if backend == "onnxruntime":
            return f"""import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

{device_snip}

print('Cargando modelo ONNXRuntime…')
tokenizer = AutoTokenizer.from_pretrained('{model_id}', use_fast=True)
model = ORTModelForCausalLM.from_pretrained('{model_id}')
model.to(device)
print('✅ Modelo cargado')

def generate(prompt: str, **gen_kwargs) -> str:
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    ids    = model.generate(**inputs, **gen_kwargs)
    return tokenizer.decode(ids[0], skip_special_tokens=True)
"""

        if backend == "vllm":
            return f"""from vllm import LLM, SamplingParams

# Configuración del modelo
print("Cargando modelo con vLLM...")
llm = LLM(model='{model_id}')
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
print("✅ Modelo cargado correctamente")

def generate(prompt: str, **gen_kwargs) -> str:
    # Caso especial para vLLM
    params = SamplingParams(
        temperature=gen_kwargs.get('temperature', 0.7),
        max_tokens=gen_kwargs.get('max_new_tokens', 256),
        top_p=gen_kwargs.get('top_p', 0.9),
        top_k=gen_kwargs.get('top_k', 40)
    )
    outputs = llm.generate([prompt], params)
    return outputs[0].outputs[0].text
"""

        if backend.startswith("transformers"):
            base = f"import torch, requests, io\nfrom transformers import pipeline, AutoTokenizer, AutoModel\n{device_snip}"
            if task == "text-generation":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para generación de texto...")
pipe = pipeline('text-generation', model='{model_id}', device=0 if torch.cuda.is_available() else -1, trust_remote_code=True)
print("✅ Modelo cargado correctamente")

def generate(prompt: str, **gen_kwargs) -> str:
    out = pipe(prompt, **gen_kwargs)
    return out[0].get('generated_text', str(out))
"""
            if task == "text-classification":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para clasificación de texto...")
pipe = pipeline('text-classification', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(text: str, **gen_kwargs) -> str:
    out = pipe(text, **gen_kwargs)
    return str(out)
"""
            if task == "token-classification":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para clasificación de tokens...")
pipe = pipeline('token-classification', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(text: str, **gen_kwargs) -> str:
    out = pipe(text, **gen_kwargs)
    return str(out)
"""
            if task == "question-answering":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para respuesta a preguntas...")
pipe = pipeline('question-answering', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(question: str, context: str, **gen_kwargs) -> str:
    out = pipe(question=question, context=context, **gen_kwargs)
    return out['answer']
"""
            if task == "summarization":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para resumen...")
pipe = pipeline('summarization', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(text: str, **gen_kwargs) -> str:
    out = pipe(text, **gen_kwargs)
    return out[0]['summary_text']
"""
            if task == "translation":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para traducción...")
pipe = pipeline('translation', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(text: str, **gen_kwargs) -> str:
    out = pipe(text, **gen_kwargs)
    return out[0]['translation_text']
"""
            if task == "image-to-text":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para imagen a texto...")
pipe = pipeline('image-to-text', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(image_path: str, **gen_kwargs) -> str:
    from PIL import Image
    image = Image.open(image_path)
    out = pipe(image, **gen_kwargs)
    return out[0]['generated_text']
"""
            if task == "text-to-image":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para texto a imagen...")
pipe = pipeline('text-to-image', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(prompt: str, **gen_kwargs) -> str:
    image = pipe(prompt, **gen_kwargs).images[0]
    image_path = "generated_image.png"
    image.save(image_path)
    return image_path
"""
            if task == "audio-to-text":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para audio a texto...")
pipe = pipeline('automatic-speech-recognition', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(audio_path: str, **gen_kwargs) -> str:
    out = pipe(audio_path, **gen_kwargs)
    return out['text']
"""
            if task == "text-to-audio":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para texto a audio...")
pipe = pipeline('text-to-speech', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(text: str, **gen_kwargs) -> str:
    audio = pipe(text, **gen_kwargs)
    audio_path = "generated_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(audio['audio'].numpy().tobytes())
    return audio_path
"""
            if task == "fill-mask":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para rellenar máscara...")
pipe = pipeline('fill-mask', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(text: str, **gen_kwargs) -> str:
    out = pipe(text, **gen_kwargs)
    return str(out)
"""
            if task == "zero-shot-classification":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para clasificación zero-shot...")
pipe = pipeline('zero-shot-classification', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(text: str, candidate_labels: List[str], **gen_kwargs) -> str:
    out = pipe(text, candidate_labels=candidate_labels, **gen_kwargs)
    return str(out)
"""
            if task == "feature-extraction":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para extracción de características...")
pipe = pipeline('feature-extraction', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(text: str, **gen_kwargs) -> str:
    out = pipe(text, **gen_kwargs)
    return str(out)
"""
            if task == "image-classification":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para clasificación de imagen...")
pipe = pipeline('image-classification', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(image_path: str, **gen_kwargs) -> str:
    from PIL import Image
    image = Image.open(image_path)
    out = pipe(image, **gen_kwargs)
    return str(out)
"""
            if task == "object-detection":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para detección de objetos...")
pipe = pipeline('object-detection', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(image_path: str, **gen_kwargs) -> str:
    from PIL import Image
    image = Image.open(image_path)
    out = pipe(image, **gen_kwargs)
    return str(out)
"""
            if task == "image-segmentation":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para segmentación de imagen...")
pipe = pipeline('image-segmentation', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(image_path: str, **gen_kwargs) -> str:
    from PIL import Image
    image = Image.open(image_path)
    out = pipe(image, **gen_kwargs)
    return str(out)
"""
            if task == "depth-estimation":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para estimación de profundidad...")
pipe = pipeline('depth-estimation', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(image_path: str, **gen_kwargs) -> str:
    from PIL import Image
    image = Image.open(image_path)
    out = pipe(image, **gen_kwargs)
    return str(out)
"""
            if task == "image-feature-extraction":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para extracción de características de imagen...")
pipe = pipeline('image-feature-extraction', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(image_path: str, **gen_kwargs) -> str:
    from PIL import Image
    image = Image.open(image_path)
    out = pipe(image, **gen_kwargs)
    return str(out)
"""
            if task == "image-to-image":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para imagen a imagen...")
pipe = pipeline('image-to-image', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(image_path: str, **gen_kwargs) -> str:
    from PIL import Image
    image = Image.open(image_path)
    out = pipe(image, **gen_kwargs).images[0]
    image_path = "generated_image.png"
    out.save(image_path)
    return image_path
"""
            if task == "text-to-video":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para texto a video...")
pipe = pipeline('text-to-video', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(prompt: str, **gen_kwargs) -> str:
    video_path = "generated_video.mp4"
    video = pipe(prompt, **gen_kwargs).images[0] # Assuming it returns a PIL Image or similar that can be saved as video
    video.save(video_path) # This might need a proper video saving library
    return video_path
"""
            if task == "video-classification":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para clasificación de video...")
pipe = pipeline('video-classification', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(video_path: str, **gen_kwargs) -> str:
    out = pipe(video_path, **gen_kwargs)
    return str(out)
"""
            if task == "audio-classification":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para clasificación de audio...")
pipe = pipeline('audio-classification', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(audio_path: str, **gen_kwargs) -> str:
    out = pipe(audio_path, **gen_kwargs)
    return str(out)
"""
            if task == "automatic-speech-recognition":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para reconocimiento automático de voz...")
pipe = pipeline('automatic-speech-recognition', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(audio_path: str, **gen_kwargs) -> str:
    out = pipe(audio_path, **gen_kwargs)
    return out['text']
"""
            if task == "speech-to-text":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para voz a texto...")
pipe = pipeline('automatic-speech-recognition', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(audio_path: str, **gen_kwargs) -> str:
    out = pipe(audio_path, **gen_kwargs)
    return out['text']
"""
            if task == "text-to-speech":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para texto a voz...")
pipe = pipeline('text-to-speech', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(text: str, **gen_kwargs) -> str:
    audio = pipe(text, **gen_kwargs)
    audio_path = "generated_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(audio['audio'].numpy().tobytes())
    return audio_path
"""
            if task == "visual-question-answering":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para respuesta a preguntas visuales...")
pipe = pipeline('visual-question-answering', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(image_path: str, question: str, **gen_kwargs) -> str:
    from PIL import Image
    image = Image.open(image_path)
    out = pipe(image=image, question=question, **gen_kwargs)
    return out[0]['answer']
"""
            if task == "document-question-answering":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para respuesta a preguntas de documentos...")
pipe = pipeline('document-question-answering', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(image_path: str, question: str, **gen_kwargs) -> str:
    from PIL import Image
    image = Image.open(image_path)
    out = pipe(image=image, question=question, **gen_kwargs)
    return out[0]['answer']
"""
            if task == "image-to-text-generation":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para generación de texto a partir de imagen...")
pipe = pipeline('image-to-text', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(image_path: str, **gen_kwargs) -> str:
    from PIL import Image
    image = Image.open(image_path)
    out = pipe(image, **gen_kwargs)
    return out[0]['generated_text']
"""
            if task == "text-to-image-generation":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para generación de imagen a partir de texto...")
pipe = pipeline('text-to-image', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(prompt: str, **gen_kwargs) -> str:
    image = pipe(prompt, **gen_kwargs).images[0]
    image_path = "generated_image.png"
    image.save(image_path)
    return image_path
"""
            if task == "text-to-audio-generation":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para generación de audio a partir de texto...")
pipe = pipeline('text-to-speech', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(text: str, **gen_kwargs) -> str:
    audio = pipe(text, **gen_kwargs)
    audio_path = "generated_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(audio['audio'].numpy().tobytes())
    return audio_path
"""
            if task == "audio-to-text-generation":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para generación de texto a partir de audio...")
pipe = pipeline('automatic-speech-recognition', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(audio_path: str, **gen_kwargs) -> str:
    out = pipe(audio_path, **gen_kwargs)
    return out['text']
"""
            if task == "text-to-video-generation":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para generación de video a partir de texto...")
pipe = pipeline('text-to-video', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")

def generate(prompt: str, **gen_kwargs) -> str:
    video_path = "generated_video.mp4"
    video = pipe(prompt, **gen_kwargs).images[0] # Assuming it returns a PIL Image or similar that can be saved as video
    video.save(video_path) # This might need a proper video saving library
    return video_path
"""

        return "" # Default case if no backend matches

    def _get_prompt_cell(self, task: str, modality: str) -> str:
        """Genera el contenido de la celda del prompt con widgets interactivos."""
        if task == "question-answering":
            return dedent("""
                import ipywidgets as widgets
                from IPython.display import display, clear_output

                question_box = widgets.Textarea(
                    value="¿Cuál es la capital de Francia?",
                    placeholder="Escribe tu pregunta aquí...",
                    description="Pregunta:",
                    layout=widgets.Layout(width='100%', height='80px')
                )

                context_box = widgets.Textarea(
                    value="París es la capital de Francia.",
                    placeholder="Escribe el contexto aquí...",
                    description="Contexto:",
                    layout=widgets.Layout(width='100%', height='120px')
                )

                generate_button = widgets.Button(
                    description="Generar Respuesta",
                    button_style='primary',
                    layout=widgets.Layout(width='200px')
                )

                output = widgets.Output()

                def on_generate_clicked(_):
                    with output:
                        clear_output()
                        print("Generando respuesta...")
                        result = generate(question=question_box.value, context=context_box.value)
                        print(f"Respuesta: {result}")

                generate_button.on_click(on_generate_clicked)
                display(question_box, context_box, generate_button, output)
                """
            )
        elif task == "zero-shot-classification":
            return dedent("""
                import ipywidgets as widgets
                from IPython.display import display, clear_output

                text_box = widgets.Textarea(
                    value="Este es un texto sobre política.",
                    placeholder="Escribe el texto a clasificar...",
                    description="Texto:",
                    layout=widgets.Layout(width='100%', height='80px')
                )

                labels_box = widgets.Textarea(
                    value="política, finanzas, deportes",
                    placeholder="Etiquetas separadas por comas...",
                    description="Etiquetas:",
                    layout=widgets.Layout(width='100%', height='60px')
                )

                generate_button = widgets.Button(
                    description="Clasificar",
                    button_style='primary',
                    layout=widgets.Layout(width='200px')
                )

                output = widgets.Output()

                def on_generate_clicked(_):
                    with output:
                        clear_output()
                        print("Clasificando texto...")
                        candidate_labels = [label.strip() for label in labels_box.value.split(',')]
                        result = generate(text=text_box.value, candidate_labels=candidate_labels)
                        print(f"Clasificación: {result}")

                generate_button.on_click(on_generate_clicked)
                display(text_box, labels_box, generate_button, output)
                """
            )
        elif task in ["image-to-text", "image-classification", "object-detection", "image-segmentation", "depth-estimation", "image-feature-extraction", "image-to-image", "visual-question-answering", "document-question-answering", "image-to-text-generation", "text-to-image-generation", "text-to-video-generation"]:
            if task in ["visual-question-answering", "document-question-answering"]:
                return dedent("""
                    import ipywidgets as widgets
                    from IPython.display import display, clear_output

                    image_upload = widgets.FileUpload(
                        accept='image/*',
                        multiple=False,
                        description="Subir imagen"
                    )

                    question_box = widgets.Textarea(
                        value="¿Qué hay en la imagen?",
                        placeholder="Escribe tu pregunta sobre la imagen...",
                        description="Pregunta:",
                        layout=widgets.Layout(width='100%', height='80px')
                    )

                    generate_button = widgets.Button(
                        description="Analizar Imagen",
                        button_style='primary',
                        layout=widgets.Layout(width='200px')
                    )

                    output = widgets.Output()

                    def on_generate_clicked(_):
                        with output:
                            clear_output()
                            if image_upload.value:
                                print("Analizando imagen...")
                                # Guardar la imagen subida
                                uploaded_file = list(image_upload.value.values())[0]
                                image_path = uploaded_file['metadata']['name']
                                with open(image_path, 'wb') as f:
                                    f.write(uploaded_file['content'])
                                
                                result = generate(image_path=image_path, question=question_box.value)
                                print(f"Respuesta: {result}")
                            else:
                                print("Por favor, sube una imagen primero.")

                    generate_button.on_click(on_generate_clicked)
                    display(image_upload, question_box, generate_button, output)
                    """
                )
            else:
                return dedent("""
                    import ipywidgets as widgets
                    from IPython.display import display, clear_output

                    image_upload = widgets.FileUpload(
                        accept='image/*',
                        multiple=False,
                        description="Subir imagen"
                    )

                    generate_button = widgets.Button(
                        description="Procesar Imagen",
                        button_style='primary',
                        layout=widgets.Layout(width='200px')
                    )

                    output = widgets.Output()

                    def on_generate_clicked(_):
                        with output:
                            clear_output()
                            if image_upload.value:
                                print("Procesando imagen...")
                                # Guardar la imagen subida
                                uploaded_file = list(image_upload.value.values())[0]
                                image_path = uploaded_file['metadata']['name']
                                with open(image_path, 'wb') as f:
                                    f.write(uploaded_file['content'])
                                
                                result = generate(image_path=image_path)
                                print(f"Resultado: {result}")
                            else:
                                print("Por favor, sube una imagen primero.")

                    generate_button.on_click(on_generate_clicked)
                    display(image_upload, generate_button, output)
                    """
                )
        elif task in ["audio-to-text", "audio-classification", "automatic-speech-recognition", "speech-to-text", "text-to-audio-generation", "audio-to-text-generation"]:
            return dedent("""
                import ipywidgets as widgets
                from IPython.display import display, clear_output

                audio_upload = widgets.FileUpload(
                    accept='audio/*',
                    multiple=False,
                    description="Subir audio"
                )

                generate_button = widgets.Button(
                    description="Procesar Audio",
                    button_style='primary',
                    layout=widgets.Layout(width='200px')
                )

                output = widgets.Output()

                def on_generate_clicked(_):
                    with output:
                        clear_output()
                        if audio_upload.value:
                            print("Procesando audio...")
                            # Guardar el archivo de audio subido
                            uploaded_file = list(audio_upload.value.values())[0]
                            audio_path = uploaded_file['metadata']['name']
                            with open(audio_path, 'wb') as f:
                                f.write(uploaded_file['content'])
                            
                            result = generate(audio_path=audio_path)
                            print(f"Resultado: {result}")
                        else:
                            print("Por favor, sube un archivo de audio primero.")

                generate_button.on_click(on_generate_clicked)
                display(audio_upload, generate_button, output)
                """
            )
        else:
            from textwrap import dedent

        return dedent("""
            import ipywidgets as widgets
            from IPython.display import display, clear_output

            prompt_box = widgets.Textarea(
                value="Escribe un poema sobre la luna.",
                placeholder="Escribe tu prompt aquí...",
                description="Prompt:",
                layout=widgets.Layout(width='100%', height='120px')
            )

            max_tokens_slider = widgets.IntSlider(
                value=100,
                min=10,
                max=500,
                step=10,
                description="Max tokens:",
                layout=widgets.Layout(width='300px')
            )

            temperature_slider = widgets.FloatSlider(
                value=0.7,
                min=0.1,
                max=2.0,
                step=0.1,
                description="Temperature:",
                layout=widgets.Layout(width='300px')
            )

            generate_button = widgets.Button(
                description="Generar",
                button_style='primary',
                layout=widgets.Layout(width='200px')
            )

            output = widgets.Output()

            def on_generate_clicked(_):
                with output:
                    clear_output()
                    print("Generando...")

                    kwargs = {
                        "prompt": prompt_box.value,
                        "temperature": temperature_slider.value
                    }

                    if self.info.get('recommended_backend') == 'llama-cpp-python':
                        kwargs["max_tokens"] = max_tokens_slider.value
                    else:
                        kwargs["max_new_tokens"] = max_tokens_slider.value

                    result = generate(**kwargs)
                    print(f"Resultado: {result}")

            generate_button.on_click(on_generate_clicked)
            display(prompt_box, max_tokens_slider, temperature_slider, generate_button, output)
            """)


    def _get_inference_cell(self, task: str, modality: str) -> str:
        """Genera el contenido de la celda de inferencia."""
        # Ya no necesitamos esta celda porque la inferencia se maneja en los widgets
        return dedent("""
            # La inferencia se maneja directamente en los widgets interactivos de la celda anterior.
            # Esta celda se mantiene por compatibilidad pero no es necesaria.
            print("✅ Widgets interactivos configurados. Usa los controles de arriba para generar contenido.")
            """
        )

    def _needs_result_processing(self, task: str, modality: str) -> bool:
        """Determina si se necesita una celda de procesamiento de resultados."""
        return task in ["text-to-image", "text-to-audio", "text-to-video", "image-to-image", "text-to-image-generation", "text-to-audio-generation", "text-to-video-generation"]

    def _get_result_processing_cell(self, task: str, modality: str) -> str:
        """Genera el contenido de la celda de procesamiento de resultados."""
        if task in ["text-to-image", "image-to-image", "text-to-image-generation"]:
            return dedent("""
                from IPython.display import Image
                Image(filename=result)
                """
            )
        elif task in ["text-to-audio", "text-to-audio-generation"]:
            return dedent("""
                from IPython.display import Audio
                Audio(result)
                """
            )
        elif task in ["text-to-video", "text-to-video-generation"]:
            return dedent("""
                from IPython.display import Video
                Video(result)
                """
            )
        return ""



def create_notebook_builder(model_info: Dict[str, Any], user: dict = None) -> NotebookBuilder:
    """
    Crea un constructor de notebooks adecuado para el modelo especificado.
    
    Args:
        model_info: Información del modelo
        
    Returns:
        NotebookBuilder: Constructor de notebooks
    """
    # Por ahora, simplemente devolvemos una instancia de NotebookBuilder
    # En el futuro, podríamos crear subclases específicas para diferentes tipos de modelos
    return NotebookBuilder(model_info, user=user)