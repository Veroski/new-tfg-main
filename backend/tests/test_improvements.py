import unittest
from pathlib import Path
import sys
import os

# Añadir el directorio del proyecto al path para poder importar los módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils.backend_selector import BackendSelector
from app.utils.format_detector import detect_format, is_quantized
from app.utils.dependency_installer import DependencyInstaller

class TestBackendSelector(unittest.TestCase):
    """Pruebas para el selector de backend."""
    
    def test_gguf_ctransformers(self):
        """Prueba la selección de ctransformers para modelos GGUF compatibles."""
        info = {
            "weight_format": "gguf",
            "model_type": "llama",
            "available_weight_files": ["model.gguf"]
        }
        selector = BackendSelector(info)
        self.assertEqual(selector.recommend(), "ctransformers")
    
    def test_gguf_llama_cpp(self):
        """Prueba la selección de llama-cpp-python para modelos GGUF genéricos."""
        info = {
            "weight_format": "gguf",
            "model_type": "unknown",
            "available_weight_files": ["model.gguf"]
        }
        selector = BackendSelector(info)
        self.assertEqual(selector.recommend(), "llama-cpp-python")
    
    def test_gptq(self):
        """Prueba la selección de auto-gptq para modelos GPTQ."""
        info = {
            "weight_format": "gptq-safetensors",
            "available_weight_files": ["model.safetensors"]
        }
        selector = BackendSelector(info)
        self.assertEqual(selector.recommend(), "auto-gptq")
    
    def test_awq(self):
        """Prueba la selección de autoawq para modelos AWQ."""
        info = {
            "weight_format": "safetensors",
            "available_weight_files": ["model.awq"]
        }
        selector = BackendSelector(info)
        self.assertEqual(selector.recommend(), "autoawq")
    
    def test_exllama(self):
        """Prueba la selección de exllama para modelos ExLlama."""
        info = {
            "weight_format": "safetensors",
            "model_id": "TheBloke/Llama-2-7B-exllama",
            "available_weight_files": ["model.safetensors"]
        }
        selector = BackendSelector(info)
        self.assertEqual(selector.recommend(), "exllama")
    
    def test_vllm(self):
        """Prueba la selección de vllm para modelos grandes no cuantizados."""
        info = {
            "modality": "text",
            "task": "text-generation",
            "param_count_estimate": 12_000_000_000,
            "quant": "none"
        }
        selector = BackendSelector(info)
        self.assertEqual(selector.recommend(), "vllm")
    
    def test_transformers_8bit(self):
        """Prueba la selección de transformers-8bit para modelos grandes."""
        info = {
            "modality": "text",
            "task": "text-generation",
            "param_count_estimate": 7_000_000_000,
            "quant": "none"
        }
        selector = BackendSelector(info)
        self.assertEqual(selector.recommend(), "transformers-8bit")
    
    def test_transformers_fallback(self):
        """Prueba el fallback a transformers cuando no hay coincidencias."""
        info = {
            "modality": "text",
            "task": "text-classification",
            "param_count_estimate": 500_000_000
        }
        selector = BackendSelector(info)
        self.assertEqual(selector.recommend(), "transformers")

class TestFormatDetector(unittest.TestCase):
    """Pruebas para el detector de formatos."""
    
    def test_gguf_detection(self):
        """Prueba la detección de formato GGUF."""
        files = ["model.gguf", "config.json"]
        self.assertEqual(detect_format(files), "gguf")
    
    def test_awq_detection(self):
        """Prueba la detección de formato AWQ."""
        files = ["model.awq", "config.json"]
        self.assertEqual(detect_format(files), "awq")
    
    def test_marlin_detection(self):
        """Prueba la detección de formato Marlin."""
        files = ["model.marlin", "config.json"]
        self.assertEqual(detect_format(files), "marlin")
    
    def test_mlc_detection(self):
        """Prueba la detección de formato MLC."""
        files = ["model.mlc", "config.json"]
        self.assertEqual(detect_format(files), "mlc")
    
    def test_exllama_detection(self):
        """Prueba la detección de formato ExLlama."""
        files = ["model-exllama.safetensors", "config.json"]
        self.assertEqual(detect_format(files), "exllama-safetensors")
    
    def test_gptq_detection(self):
        """Prueba la detección de formato GPTQ."""
        files = ["model-gptq.safetensors", "config.json"]
        self.assertEqual(detect_format(files), "gptq-safetensors")
    
    def test_is_quantized(self):
        """Prueba la detección de cuantización."""
        self.assertEqual(is_quantized("gptq-safetensors"), "gptq")
        self.assertEqual(is_quantized("gguf"), "gguf-4bit")
        self.assertEqual(is_quantized("awq"), "awq")
        self.assertEqual(is_quantized("exllama-safetensors"), "exllama")
        self.assertEqual(is_quantized("marlin"), "marlin-4bit")
        self.assertEqual(is_quantized("safetensors"), "none")

class TestDependencyInstaller(unittest.TestCase):
    """Pruebas para el instalador de dependencias."""
    
    def test_transformers_commands(self):
        """Prueba los comandos para transformers."""
        installer = DependencyInstaller()
        cmds = installer.get_install_commands("transformers", "text")
        self.assertEqual(cmds, ["pip install -q --upgrade transformers"])
    
    def test_transformers_vision_commands(self):
        """Prueba los comandos para transformers con visión."""
        installer = DependencyInstaller()
        cmds = installer.get_install_commands("transformers", "vision")
        self.assertEqual(cmds, ["pip install -q --upgrade transformers[vision]"])
    
    def test_transformers_8bit_commands(self):
        """Prueba los comandos para transformers-8bit."""
        installer = DependencyInstaller()
        cmds = installer.get_install_commands("transformers-8bit", "text")
        self.assertEqual(cmds, ["pip install -q --upgrade transformers", "pip install -q bitsandbytes"])
    
    def test_llama_cpp_commands(self):
        """Prueba los comandos para llama-cpp-python."""
        installer = DependencyInstaller()
        cmds = installer.get_install_commands("llama-cpp-python")
        self.assertEqual(cmds, ["pip install -q llama-cpp-python huggingface-hub"])
    
    def test_autoawq_commands(self):
        """Prueba los comandos para autoawq."""
        installer = DependencyInstaller()
        cmds = installer.get_install_commands("autoawq")
        self.assertEqual(cmds, ["pip install -q autoawq accelerate"])
    
    def test_vllm_commands(self):
        """Prueba los comandos para vllm."""
        installer = DependencyInstaller()
        cmds = installer.get_install_commands("vllm")
        self.assertEqual(cmds, ["pip install -q vllm"])
    
    def test_unknown_backend(self):
        """Prueba el comportamiento con un backend desconocido."""
        installer = DependencyInstaller()
        cmds = installer.get_install_commands("unknown_backend")
        self.assertEqual(cmds, [])

if __name__ == '__main__':
    unittest.main()
