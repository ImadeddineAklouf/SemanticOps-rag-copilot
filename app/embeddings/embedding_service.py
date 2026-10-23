from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingService:
    """Multilingual sentence embeddings.

    The model is loaded lazily on first use, so importing the application
    stays fast and does not require the model to be downloaded.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        return self.model.encode(
            text, normalize_embeddings=True
        ).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts, normalize_embeddings=True
        ).tolist()
