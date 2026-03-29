from bookworm.base import Embedder


class EmbeddingService:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.embedder.embed(texts)
        return embeddings
