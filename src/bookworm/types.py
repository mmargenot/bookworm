import uuid
from pydantic import BaseModel, Field
from enum import Enum


class EmbeddingType(str, Enum):
    TEXT = "text"
    MULTIMODAL = "multimodal"


class Source(BaseModel):
    pass


class Document(BaseModel):
    """
    Base class for unstructured data.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    parent_id: str | None = None
    name: str
    source: Source
    content: str


class Embedding(BaseModel):
    """ """

    source_id: str
    type: EmbeddingType
    embedding: list[float]


class Cluster(BaseModel):
    """
    Collection of documents, grouped with clustering methods.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    label: int
    document_ids: list[str]
    parent_id: str | None
