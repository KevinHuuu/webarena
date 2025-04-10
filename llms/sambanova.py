import os
import openai
from typing import List, Dict, Any, Optional

class SambaNovaClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.sambanova.ai/v1",
        model: str = "Meta-Llama-3.3-70B-Instruct",
        temperature: float = 0.1,
        top_p: float = 0.1
    ):
        self.api_key = api_key or os.environ.get("SAMBANOVA_API_KEY")
        if not self.api_key:
            raise ValueError("SambaNova API key not provided")
            
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )
        self.model = model
        self.temperature = temperature
        self.top_p = top_p

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            top_p=top_p or self.top_p
        )
        return response.choices[0].message.content

    def __call__(self, prompt: str, system_message: str = "You are a helpful assistant") -> str:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        return self.generate(messages) 