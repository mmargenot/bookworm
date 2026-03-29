import logging

from bookworm.base import Embedder


logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def embed(self, texts: list[str]) -> list[list[float]]:
        num_texts = len(texts)
        logger.info(f"Embedding {num_texts} texts")
        embeddings = self.embedder.embed(texts)
        embed_dim = len(embeddings[0])
        logger.info(f"Built {num_texts} embeddings of dim {embed_dim}")
        return embeddings
