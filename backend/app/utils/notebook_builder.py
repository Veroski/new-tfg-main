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
    
    def __init__(self, info: Dict[str, Any]):
        """
        Inicializa el constructor de notebooks.
        
        Args:
            info: Diccionario con información del modelo
        """
        self.nb = nbf.v4.new_notebook()
        self.info = info
    
    def build(self) -> nbf.NotebookNode:
        """
        Construye el notebook completo.
        
        Returns:
            nbf.NotebookNode: Notebook generado
        """
        self.add_metadata()
        self.add_install()
        self.add_gpu_check()
        self.add_download_weights()
        self.add_readme_preview()
        self.add_model_setup()
        self.add_prompt_cell()
        self.add_inference_cell()
        self.add_result_processing_cell()
        self.add_sample_eval()
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
    
    def add_sample_eval(self) -> None:
        """Añade la celda de evaluación de muestra."""
        eval_cell = self._sample_eval_cell(
            self.info.get('task', ''), 
            self.info.get('model_id', '')
        )
        if eval_cell:  # Solo añadir si hay contenido
            self.nb.cells.append(nbf.v4.new_code_cell(eval_cell))
    
    def add_footer(self) -> None:
        """Añade la celda de pie de página."""
        self.nb.cells.append(
            nbf.v4.new_markdown_cell(
                "---\n✅ *Notebook generado automáticamente. ¡Disfruta!* / *Happy hacking!*"
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
"""

        if backend == "vllm":
            return f"""from vllm import LLM, SamplingParams

# Configuración del modelo
print("Cargando modelo con vLLM...")
llm = LLM(model='{model_id}')
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
print("✅ Modelo cargado correctamente")
"""

        if backend.startswith("transformers"):
            base = f"import torch, requests, io\nfrom transformers import pipeline, AutoTokenizer, AutoModel\n{device_snip}"
            if task == "text-generation":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para generación de texto...")
pipe = pipeline('text-generation', model='{model_id}', device=0 if torch.cuda.is_available() else -1, trust_remote_code=True)
print("✅ Modelo cargado correctamente")
"""
            if task == "text-classification":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para clasificación de texto...")
pipe = pipeline('text-classification', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")
"""
            if task == "image-classification":
                return base + f"""
from PIL import Image

# Configuración del modelo
print("Cargando modelo para clasificación de imágenes...")
pipe = pipeline('image-classification', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")
"""
            if task == "automatic-speech-recognition":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para reconocimiento de voz...")
pipe = pipeline('automatic-speech-recognition', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")
"""
            if task == "image-to-text" or task == "image-captioning":
                return base + f"""
from PIL import Image

# Configuración del modelo
print("Cargando modelo para descripción de imágenes...")
pipe = pipeline('{task}', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")
"""
            if task == "audio-classification":
                return base + f"""

# Configuración del modelo
print("Cargando modelo para clasificación de audio...")
pipe = pipeline('audio-classification', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")
"""
            if task == "text-to-audio":
                return base + f"""
import IPython.display as ipd

# Configuración del modelo
print("Cargando modelo para generación de audio...")
pipe = pipeline('text-to-audio', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")
"""
            if task == "video-classification":
                return base + f"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Configuración del modelo
print("Cargando modelo para clasificación de video...")
pipe = pipeline('video-classification', model='{model_id}', device=0 if torch.cuda.is_available() else -1)
print("✅ Modelo cargado correctamente")
"""
            # genérico
            return base + f"""

# Configuración del modelo
print("Cargando modelo para la tarea '{task}'...")
pipe = pipeline('{task}', model='{model_id}', device=0 if torch.cuda.is_available() else -1, trust_remote_code=True)
print("✅ Modelo cargado correctamente")
"""

        if backend == "diffusers":
            return f"""import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# Configurar el dispositivo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando dispositivo: {{device}}")

# Configuración del modelo
print("Cargando modelo de difusión...")
pipe = StableDiffusionPipeline.from_pretrained('{model_id}', torch_dtype=torch.float16 if device == 'cuda' else torch.float32)
pipe = pipe.to(device)
print("✅ Modelo cargado correctamente")
"""

        # Fallback para otros backends
        return f"""# Configuración del modelo para backend '{backend}' con modalidad '{modality}'

import torch
print("Dispositivo disponible:", 'cuda' if torch.cuda.is_available() else 'cpu')

print("Modelo ID:", '{model_id}')
print("Tarea:", '{task}')
print("Modalidad:", '{modality}')
print("Backend recomendado:", '{backend}')

# Añade aquí tu código específico para este modelo
"""

    def _get_prompt_cell(self, task: str, modality: str) -> str:
        """Genera el contenido de la celda de prompt."""
        if task == "text-generation":
            return dedent("""
            # Define tu prompt aquí
            prompt = "Explica de manera sencilla qué es la inteligencia artificial y cómo está cambiando el mundo"
            
            # Puedes modificar este prompt sin necesidad de volver a cargar el modelo
            print(f"Prompt: {prompt}")
            """)
        
        elif task == "text-classification":
            return dedent("""
            # Define los textos a clasificar aquí
            texts = [
                'I love this product! It exceeded all my expectations and I would definitely buy it again.',
                'This movie was terrible. The plot made no sense and the acting was awful.',
                'The restaurant was okay. Food was good but the service was slow.',
                'I am absolutely thrilled with my purchase! Best decision ever!'
            ]
            
            # Puedes modificar estos textos sin necesidad de volver a cargar el modelo
            print(f"Textos a clasificar: {len(texts)}")
            """)
        
        elif task == "image-classification":
            return dedent("""
            # Define las URLs de las imágenes a clasificar
            image_urls = [
                'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg',
                'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/mountain.jpg'
            ]
            
            # Carga las imágenes
            from PIL import Image
            from io import BytesIO
            import requests
            
            images = []
            for url in image_urls:
                print(f"Cargando imagen desde {url}")
                img = Image.open(BytesIO(requests.get(url).content))
                images.append(img)
                
            print(f"Imágenes cargadas: {len(images)}")
            """)
        
        elif task == "automatic-speech-recognition":
            return dedent("""
            # Define las URLs de los audios a transcribir
            audio_urls = [
                'https://huggingface.co/datasets/Narsil/audio_dummy/resolve/main/1.flac',
                'https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/resolve/main/es/clips/common_voice_es_19362189.mp3'
            ]
            
            # Puedes modificar estas URLs sin necesidad de volver a cargar el modelo
            print(f"URLs de audio a transcribir: {len(audio_urls)}")
            """)
        
        elif task == "image-to-text" or task == "image-captioning":
            return dedent("""
            # Define las URLs de las imágenes a describir
            image_urls = [
                'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg',
                'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/mountain.jpg'
            ]
            
            # Carga las imágenes
            from PIL import Image
            from io import BytesIO
            import requests
            
            images = []
            for url in image_urls:
                print(f"Cargando imagen desde {url}")
                img = Image.open(BytesIO(requests.get(url).content))
                images.append(img)
                
            print(f"Imágenes cargadas: {len(images)}")
            """)
        
        elif task == "text-to-image":
            return dedent("""
            # Define los prompts para generar imágenes
            image_prompts = [
                "a photograph of an orange cat sitting on a windowsill, looking outside, high quality, detailed",
                "a beautiful mountain landscape with a lake at sunset, photorealistic, high resolution"
            ]
            
            # Puedes modificar estos prompts sin necesidad de volver a cargar el modelo
            print(f"Prompts para generar imágenes: {len(image_prompts)}")
            """)
        
        elif task == "audio-classification":
            return dedent("""
            # Define las URLs de los audios a clasificar
            audio_urls = [
                'https://huggingface.co/datasets/Narsil/audio_dummy/resolve/main/1.flac',
                'https://huggingface.co/datasets/sanchit-gandhi/librispeech_asr_dummy/resolve/main/1.flac'
            ]
            
            # Puedes modificar estas URLs sin necesidad de volver a cargar el modelo
            print(f"URLs de audio a clasificar: {len(audio_urls)}")
            """)
        
        elif task == "text-to-audio":
            return dedent("""
            # Define los prompts para generar audio
            audio_prompts = [
                "Una voz femenina diciendo: Bienvenidos a este tutorial de inteligencia artificial",
                "A male voice saying: Artificial intelligence is transforming the world"
            ]
            
            # Puedes modificar estos prompts sin necesidad de volver a cargar el modelo
            print(f"Prompts para generar audio: {len(audio_prompts)}")
            """)
        
        elif task == "video-classification":
            return dedent("""
            # Función para obtener frames de ejemplo (simulando un video)
            def get_sample_frames(num_frames=8):
                # URLs de ejemplo (podríamos usar frames reales de un video)
                urls = [
                    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg',
                    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/mountain.jpg'
                ]
                
                import requests
                from PIL import Image
                from io import BytesIO
                import numpy as np
                
                frames = []
                for i in range(num_frames):
                    url = urls[i % len(urls)]
                    img = Image.open(BytesIO(requests.get(url).content)).resize((224, 224))
                    frames.append(np.array(img))
                
                return frames
            
            # Obtener frames de muestra
            print("Obteniendo frames de muestra para clasificación de video...")
            frames = get_sample_frames()
            print(f"Frames obtenidos: {len(frames)}")
            """)
        
        # Genérico para otros casos
        return dedent("""
        # Define tu input aquí según el tipo de modelo
        input_data = "Ejemplo de entrada para el modelo"
        
        # Puedes modificar este input sin necesidad de volver a cargar el modelo
        print(f"Input: {input_data}")
        """)

    def _get_inference_cell(self, task: str, modality: str) -> str:
        """Genera el contenido de la celda de inferencia."""
        if task == "text-generation":
            return dedent("""
            # Ejecutar inferencia con el prompt
            try:
                # Diferentes backends pueden requerir diferentes formas de llamada
                try:
                    # Para transformers pipeline
                    result = pipe(prompt, max_new_tokens=256)
                    if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
                        generated_text = result[0]['generated_text']
                    else:
                        generated_text = str(result)
                except:
                    # Para otros backends (llama-cpp, ctransformers, etc.)
                    try:
                        # Intento para llama-cpp
                        result = llm(prompt, max_tokens=256)
                        generated_text = result["choices"][0]["text"]
                    except:
                        # Intento para otros backends
                        generated_text = llm(prompt, max_new_tokens=256)
                
                print("\\nResultado de la generación:")
                print("-" * 50)
                print(generated_text)
                print("-" * 50)
            except Exception as e:
                print(f"Error durante la inferencia: {e}")
                print("Intenta ajustar el código según el backend específico del modelo.")
            """)
        
        elif task == "text-classification":
            return dedent("""
            # Ejecutar inferencia con los textos
            try:
                results = []
                for i, text in enumerate(texts):
                    result = pipe(text)
                    results.append(result)
                    print(f"\\nTexto {i+1}: '{text}'")
                    print(f"Clasificación: {result[0]['label']} (Score: {result[0]['score']:.4f})")
            except Exception as e:
                print(f"Error durante la inferencia: {e}")
            """)
        
        elif task == "image-classification":
            return dedent("""
            # Ejecutar inferencia con las imágenes
            try:
                for i, img in enumerate(images):
                    result = pipe(img)
                    print(f"\\nResultados para imagen {i+1}:")
                    for j, res in enumerate(result[:5]):  # Mostrar los 5 primeros resultados
                        print(f"  {j+1}. {res['label']} ({res['score']:.4f})")
                    
                    # Mostrar la imagen
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(6, 6))
                    plt.imshow(img)
                    plt.title(f"Imagen {i+1}: {result[0]['label']}")
                    plt.axis('off')
                    plt.show()
            except Exception as e:
                print(f"Error durante la inferencia: {e}")
            """)
        
        elif task == "automatic-speech-recognition":
            return dedent("""
            # Ejecutar inferencia con los audios
            try:
                results = []
                for i, url in enumerate(audio_urls):
                    print(f"\\nTranscribiendo audio {i+1}...")
                    result = pipe(url)
                    results.append(result)
                    print(f"Transcripción: {result['text']}")
                    
                    # Reproducir el audio
                    import IPython.display as ipd
                    print(f"Reproduciendo audio {i+1}:")
                    ipd.display(ipd.Audio(url))
            except Exception as e:
                print(f"Error durante la inferencia: {e}")
            """)
        
        elif task == "image-to-text" or task == "image-captioning":
            return dedent("""
            # Ejecutar inferencia con las imágenes
            try:
                results = []
                for i, img in enumerate(images):
                    result = pipe(img)
                    results.append(result)
                    caption = result[0]['generated_text']
                    print(f"\\nDescripción para imagen {i+1}: {caption}")
                    
                    # Mostrar la imagen con su descripción
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(8, 8))
                    plt.imshow(img)
                    plt.title(caption)
                    plt.axis('off')
                    plt.show()
            except Exception as e:
                print(f"Error durante la inferencia: {e}")
            """)
        
        elif task == "text-to-image":
            return dedent("""
            # Ejecutar inferencia con los prompts
            try:
                generated_images = []
                for i, prompt in enumerate(image_prompts):
                    print(f"\\nGenerando imagen {i+1} con prompt: '{prompt}'")
                    result = pipe(prompt)
                    image = result.images[0]
                    generated_images.append(image)
                    
                    # Mostrar la imagen generada
                    display(image)
                    
                    # Guardar la imagen
                    filename = f"generated_image_{i+1}.png"
                    image.save(filename)
                    print(f"Imagen guardada como '{filename}'")
            except Exception as e:
                print(f"Error durante la inferencia: {e}")
            """)
        
        elif task == "audio-classification":
            return dedent("""
            # Ejecutar inferencia con los audios
            try:
                results = []
                for i, url in enumerate(audio_urls):
                    print(f"\\nClasificando audio {i+1}...")
                    result = pipe(url)
                    results.append(result)
                    print("Resultados:")
                    for j, res in enumerate(result[:5]):  # Mostrar los 5 primeros resultados
                        print(f"  {j+1}. {res['label']} ({res['score']:.4f})")
                    
                    # Reproducir el audio
                    import IPython.display as ipd
                    print(f"Reproduciendo audio {i+1}:")
                    ipd.display(ipd.Audio(url))
            except Exception as e:
                print(f"Error durante la inferencia: {e}")
            """)
        
        elif task == "text-to-audio":
            return dedent("""
            # Ejecutar inferencia con los prompts
            try:
                results = []
                for i, prompt in enumerate(audio_prompts):
                    print(f"\\nGenerando audio {i+1} con prompt: '{prompt}'")
                    result = pipe(prompt)
                    results.append(result)
                    
                    # Extraer información del audio
                    sampling_rate = result.get("sampling_rate", 16000)
                    audio_array = result.get("audio", result.get("audio_array", None))
                    
                    if audio_array is not None:
                        # Reproducir el audio generado
                        import IPython.display as ipd
                        print(f"Reproduciendo audio generado {i+1}:")
                        ipd.display(ipd.Audio(audio_array, rate=sampling_rate))
                    else:
                        print("No se pudo obtener el array de audio del resultado")
            except Exception as e:
                print(f"Error durante la inferencia: {e}")
            """)
        
        elif task == "video-classification":
            return dedent("""
            # Ejecutar inferencia con los frames
            try:
                # Mostrar algunos frames
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 4, figsize=(15, 5))
                for i, ax in enumerate(axes):
                    ax.imshow(frames[i])
                    ax.set_title(f"Frame {i+1}")
                    ax.axis('off')
                plt.tight_layout()
                plt.show()
                
                # Clasificar el "video"
                print("\\nClasificando video...")
                result = pipe(frames)
                print("Resultados:")
                for i, res in enumerate(result[:5]):  # Mostrar los 5 primeros resultados
                    print(f"  {i+1}. {res['label']} ({res['score']:.4f})")
            except Exception as e:
                print(f"Error durante la inferencia: {e}")
            """)
        
        # Genérico para otros casos
        return dedent("""
        # Ejecutar inferencia con el input
        try:
            result = pipe(input_data)
            print("\\nResultado de la inferencia:")
            print(result)
        except Exception as e:
            print(f"Error durante la inferencia: {e}")
            print("Intenta ajustar el código según el tipo específico de modelo.")
        """)

    def _needs_result_processing(self, task: str, modality: str) -> bool:
        """Determina si se necesita una celda de procesamiento de resultados."""
        # Tareas que generan resultados que se pueden guardar
        save_result_tasks = [
            "text-to-audio", "audio-to-audio", "text-to-image", 
            "image-to-image", "text-to-video", "automatic-speech-recognition"
        ]
        
        return task in save_result_tasks or modality in ["audio", "video"]

    def _get_result_processing_cell(self, task: str, modality: str) -> str:
        """Genera el contenido de la celda de procesamiento de resultados."""
        if task == "text-to-audio" or task == "audio-to-audio" or modality == "audio":
            return dedent("""
            # Procesar y guardar el resultado de audio
            import numpy as np
            from scipy.io.wavfile import write
            import IPython.display as ipd

            # Seleccionar el resultado a guardar (por defecto el primero)
            result_index = 0  # Cambia esto si quieres guardar otro resultado
            
            if len(results) > result_index:
                result = results[result_index]
                
                # Validar sample rate
                sampling_rate = int(result.get("sampling_rate", 16000))
                if not (0 <= sampling_rate <= 65535):
                    print(f"Advertencia: Sampling rate fuera de rango: {sampling_rate}, usando 16000 por defecto")
                    sampling_rate = 16000
                
                # Validar y normalizar audio
                audio_data = result.get("audio", result.get("audio_array", None))
                if audio_data is not None:
                    if not isinstance(audio_data, np.ndarray):
                        audio_data = np.array(audio_data)
                    
                    # Asegurar que sea 1D y de tipo float
                    if audio_data.ndim > 1:
                        audio_data = audio_data.squeeze()
                    
                    if np.issubdtype(audio_data.dtype, np.floating):
                        # Escalar a int16
                        max_val = np.max(np.abs(audio_data))
                        if max_val > 0:
                            audio_data = audio_data / max_val  # normaliza a [-1, 1]
                        audio_data = (audio_data * 32767).astype(np.int16)
                    
                    # Guardar el audio
                    output_filename = "output_audio.wav"
                    write(output_filename, rate=sampling_rate, data=audio_data)
                    print(f"✅ Audio guardado como '{output_filename}'")
                    
                    # Reproducir el audio guardado
                    print("Reproduciendo audio guardado:")
                    ipd.display(ipd.Audio(output_filename))
                else:
                    print("No se pudo obtener datos de audio del resultado")
            else:
                print("No hay resultados disponibles para guardar")
            """)
        
        elif task == "text-to-image" or task == "image-to-image":
            return dedent("""
            # Procesar y guardar el resultado de imagen
            
            # Seleccionar la imagen a guardar (por defecto la primera)
            image_index = 0  # Cambia esto si quieres guardar otra imagen
            
            if len(generated_images) > image_index:
                image = generated_images[image_index]
                
                # Guardar la imagen en alta resolución
                output_filename = "output_image_hires.png"
                image.save(output_filename)
                print(f"✅ Imagen guardada en alta resolución como '{output_filename}'")
                
                # Mostrar la imagen guardada
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                plt.axis('off')
                plt.title("Imagen guardada")
                plt.show()
            else:
                print("No hay imágenes disponibles para guardar")
            """)
        
        elif task == "text-to-video" or modality == "video":
            return dedent("""
            # Procesar y guardar el resultado de video
            
            # Nota: Este código asume que el resultado es una lista de frames o un objeto con atributo 'frames'
            try:
                # Intentar obtener los frames del resultado
                if hasattr(result, 'frames'):
                    frames = result.frames
                elif isinstance(result, list) and len(result) > 0 and hasattr(result[0], 'frames'):
                    frames = result[0].frames
                elif isinstance(result, list) and all(isinstance(f, np.ndarray) for f in result):
                    frames = result
                else:
                    frames = None
                    print("No se pudieron identificar los frames en el resultado")
                
                if frames is not None and len(frames) > 0:
                    # Guardar como video
                    try:
                        from diffusers.utils import export_to_video
                        output_filename = "output_video.mp4"
                        export_to_video(frames, output_filename)
                        print(f"✅ Video guardado como '{output_filename}'")
                        
                        # Mostrar el video
                        import IPython.display as ipd
                        ipd.display(ipd.Video(output_filename, width=350))
                    except Exception as e:
                        print(f"Error al guardar el video: {e}")
                        
                        # Alternativa: guardar frames individuales
                        print("Guardando frames individuales...")
                        import matplotlib.pyplot as plt
                        for i, frame in enumerate(frames[:10]):  # Guardar solo los primeros 10 frames
                            frame_filename = f"output_frame_{i:03d}.png"
                            plt.imsave(frame_filename, frame)
                        print(f"✅ {min(10, len(frames))} frames guardados como archivos PNG")
            except Exception as e:
                print(f"Error al procesar el resultado de video: {e}")
            """)
        
        elif task == "automatic-speech-recognition":
            return dedent("""
            # Procesar y guardar el resultado de transcripción
            
            # Seleccionar el resultado a guardar (por defecto el primero)
            result_index = 0  # Cambia esto si quieres guardar otro resultado
            
            if len(results) > result_index:
                result = results[result_index]
                
                # Extraer el texto transcrito
                if isinstance(result, dict) and 'text' in result:
                    transcription = result['text']
                else:
                    transcription = str(result)
                
                # Guardar la transcripción en un archivo
                output_filename = "transcription.txt"
                with open(output_filename, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                print(f"✅ Transcripción guardada como '{output_filename}'")
                
                # Mostrar la transcripción
                print("\\nTranscripción guardada:")
                print("-" * 50)
                print(transcription)
                print("-" * 50)
            else:
                print("No hay resultados disponibles para guardar")
            """)
        
        # Genérico para otros casos
        return dedent("""
        # Procesar y guardar el resultado
        
        try:
            # Guardar el resultado en un archivo de texto
            output_filename = "output_result.txt"
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(str(result))
            print(f"✅ Resultado guardado como '{output_filename}'")
            
            # Mostrar un resumen del resultado
            print("\\nResumen del resultado guardado:")
            print("-" * 50)
            result_str = str(result)
            print(result_str[:500] + "..." if len(result_str) > 500 else result_str)
            print("-" * 50)
        except Exception as e:
            print(f"Error al guardar el resultado: {e}")
        """)

    def _sample_eval_cell(self, task: str, model_id: str) -> str:
        """Genera el contenido de la celda de evaluación de muestra."""
        # Ya no necesitamos esta celda porque hemos separado la funcionalidad
        # en celdas más específicas (prompt, inferencia, procesamiento)
        return ""


class TextNotebookBuilder(NotebookBuilder):
    """Especialización para modelos de texto."""
    pass


class VisionNotebookBuilder(NotebookBuilder):
    """Especialización para modelos de visión."""
    pass


class AudioNotebookBuilder(NotebookBuilder):
    """Especialización para modelos de audio."""
    pass


def create_notebook_builder(info: Dict[str, Any]) -> NotebookBuilder:
    """
    Factory para crear el constructor de notebooks adecuado según la modalidad.
    
    Args:
        info: Diccionario con información del modelo
        
    Returns:
        NotebookBuilder: Constructor de notebooks especializado
    """
    modality = info.get('modality', 'text')
    
    if modality == 'vision':
        return VisionNotebookBuilder(info)
    elif modality == 'audio':
        return AudioNotebookBuilder(info)
    else:
        return TextNotebookBuilder(info)