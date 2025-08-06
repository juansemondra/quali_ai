import subprocess
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

class LLMOrchestrator:
    def __init__(
        self,
        model_name: str = "deepseek-r1:8b",
        use_ollama: bool = False,
        quantize: bool = True,
    ):
        """
        Inicializa el orquestador para usar Ollama CLI o Hugging Face.

        :param model_name: nombre del modelo en Ollama o Hugging Face
        :param use_ollama: si True, usa la CLI de Ollama; si False, usa Transformers
        :param quantize: aplica cuantización 4-bit en Transformers
        """
        self.model_name = model_name
        self.use_ollama = use_ollama

        if not use_ollama:
            # setup Transformers model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            load_kwargs = {"device_map": "auto"}
            if quantize:
                load_kwargs.update({
                    "load_in_4bit": True,
                    "quantization_config": {"bnb_4bit_compute_dtype": torch.float16}
                })
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            self.gen_config = GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=256,
                pad_token_id=self.tokenizer.eos_token_id
            )

    def generate(self, prompt: str, **gen_kwargs) -> str:
        """
        Genera texto a partir de un prompt, usando Ollama o Transformers.

        :param prompt: texto de entrada para el modelo
        :param gen_kwargs: parámetros adicionales de generación para Transformers
        :return: cadena generada
        """
        if self.use_ollama:
            cmd = [
                "ollama", "generate", self.model_name,
                "--json", "--prompt", prompt
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
                return proc.stdout.strip()
            except subprocess.CalledProcessError as e:
                # Si Ollama falla, registramos y devolvemos un JSON vacío
                print(f"⚠️ Ollama error ({e.returncode}): {e.stderr}\\nPrompt: {prompt}")
                return '{"codes": [], "sentiment": "unknown"}'
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            config = self.gen_config.copy_merge(gen_kwargs)
            out = self.model.generate(**inputs, generation_config=config)
            return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def extract_codes_and_sentiment(self, fragment: str) -> dict:
        """
        Extrae códigos temáticos y sentimiento de un fragmento de texto.

        :param fragment: fragmento de entrevista
        :return: dict con claves 'codes' y 'sentiment'
        """
        prompt = (
            "Eres un asistente que, dado un fragmento de entrevista, extrae:\n"
            "1) Una lista de códigos temáticos (solo palabras clave)\n"
            "2) El sentimiento general (positivo, negativo o neutral)\n\n"
            f"Texto:\n{fragment}\n\n"
            "Responde en JSON con claves 'codes' y 'sentiment'."
        )
        raw = self.generate(prompt)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}

# Ejemplo de uso directo
if __name__ == "__main__":
    orch = LLMOrchestrator(model_name="deepseek-r1:8b", use_ollama=True)
    sample = "Durante mi transición, me sentí acogido en la escuela, pero hubo momentos de rechazo."
    result = orch.extract_codes_and_sentiment(sample)
    print(json.dumps(result, ensure_ascii=False, indent=2))
