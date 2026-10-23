import os

from groq import Groq

SYSTEM_PROMPT = (
    "Tu es un assistant qui répond uniquement à partir du contexte fourni. "
    "Si l'information n'est pas dans le contexte, dis clairement que tu ne sais pas. "
    "Réponds dans la langue de la question."
)


class LLMService:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        timeout: float = 60.0,
    ):
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY manquante : définis-la dans ton .env")

        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = Groq(api_key=api_key, timeout=timeout)

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as exc:
            raise RuntimeError(f"Erreur Groq ({self.model}) : {exc}") from exc

        return (response.choices[0].message.content or "").strip()