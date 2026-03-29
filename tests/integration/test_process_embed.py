import pytest

from bookworm.processing import DocumentProcessor
from bookworm.embedding import BGEM3Embedder
from bookworm.types import Document, Source


ANNABEL_LEE = (
    "It was many and many a year ago, in a kingdom by the sea, "
    "that a maiden there lived whom you may know by the name of Annabel Lee; "
    "and this maiden she lived with no other thought than to love and be loved by me. "
    "I was a child and she was a child, in this kingdom by the sea, "
    "but we loved with a love that was more than love — I and my Annabel Lee — "
    "with a love that the wingèd seraphs of Heaven coveted her and me. "
    "And this was the reason that, long ago, in this kingdom by the sea, "
    "a wind blew out of a cloud, chilling my beautiful Annabel Lee; "
    "so that her highborn kinsmen came and bore her away from me, "
    "to shut her up in a sepulchre in this kingdom by the sea."
)


@pytest.fixture(scope="module")
def embedder():
    return BGEM3Embedder()


@pytest.fixture(scope="module")
def processor():
    return DocumentProcessor(chunk_size=200, chunk_overlap=20)


@pytest.fixture(scope="module")
def document():
    return Document(name="annabel_lee", source=Source(), content=ANNABEL_LEE)


@pytest.fixture(scope="module")
def chunks(processor, document):
    return processor.process(document)


class TestProcessThenEmbed:
    def test_embedding_count_matches_chunks(self, embedder, chunks):
        embeddings = embedder.embed([c.content for c in chunks])
        assert len(embeddings) == len(chunks)

    def test_embedding_dimensions_consistent(self, embedder, chunks):
        embeddings = embedder.embed([c.content for c in chunks])
        dims = {len(e) for e in embeddings}
        assert len(dims) == 1

    def test_chunks_have_parent_ids(self, chunks, document):
        for chunk in chunks:
            assert chunk.parent_id == document.id
