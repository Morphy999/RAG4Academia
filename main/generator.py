import ollama
from typing import Optional
import abc
from fastapi.responses import StreamingResponse


class Generator(abc.ABC):
    
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abc.abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        pass
    
    
class OllamaGenerator(Generator):
    def __init__(self, model_name: str = "llama3"):
        super().__init__(model_name=model_name)

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None):
        for chunk in ollama.chat(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        ):
            yield chunk["message"]["content"]

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        return "".join(self.generate_stream(prompt))

