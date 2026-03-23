from bookworm.base import Embedder
from sentence_transformers import SentenceTransformer


class BGEM3Embedder(Embedder):
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-m3")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("Must submit texts")
        if any(text == "" for text in texts):
            raise ValueError("Texts must not contain empty strings")
        embeddings = self.model.encode(texts)
        # TODO: handle sparse and dense from bge-m3
        return embeddings.tolist()
