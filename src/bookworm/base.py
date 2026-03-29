"""
Flow:
    1. Generate embeddings of the texts
    2. Cluster the embeddings of the texts, map texts to clusters
    4. Cluster the clusters

Similar to RAPTOR, and runs until we have the desired number of entry clusters.
"""

from abc import ABC, abstractmethod
from bookworm.types import Cluster, Embedding, Document


class InsightModel(ABC):
    @abstractmethod
    def generate_clusters(self):
        pass


class ClusteringMethod(ABC):
    @abstractmethod
    def cluster(self, items: list[Embedding]) -> list[Cluster]:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "ClusteringMethod":
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        pass


class Reducer(ABC):
    @abstractmethod
    def reduce(self, vectors: list[list[float]]) -> list[list[float]]:
        pass


class Processor(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[Document]:
        pass
