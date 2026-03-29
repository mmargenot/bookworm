import pytest

from bookworm.processing import DocumentProcessor
from bookworm.types import Document, Source


class TestDocumentProcessor:
    @pytest.mark.parametrize(
        "chunk_size, chunk_overlap",
        [
            pytest.param(256, 512, id="Overlap larger than chunk size"),
            pytest.param(256, 256, id="Overlap matches chunk size"),
        ],
    )
    def test_invalid_inputs(self, chunk_size: int, chunk_overlap: int):
        with pytest.raises(ValueError):
            DocumentProcessor(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )

    def _make_doc(self, content: str, name: str = "test_doc") -> Document:
        return Document(name=name, source=Source(), content=content)

    def test_short_text_produces_single_chunk(self):
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        chunks = processor.process(self._make_doc("Short."))
        assert len(chunks) == 1

    def test_long_text_produces_expected_chunks(self):
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        chunks = processor.process(self._make_doc("word " * 200))
        assert len(chunks) == 12

    def test_chunk_naming(self):
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        doc = self._make_doc("word " * 200, name="my_doc")
        chunks = processor.process(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.name == f"my_doc-chunk-{i}"

    def test_parent_id_linkage(self):
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        doc = self._make_doc("word " * 200)
        chunks = processor.process(doc)
        for chunk in chunks:
            assert chunk.parent_id == doc.id

    def test_content_preserved(self):
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        original = "word " * 200
        chunks = processor.process(self._make_doc(original))
        combined = "".join(c.content for c in chunks)
        for word in original.split():
            assert word in combined

    def test_empty_content(self):
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        chunks = processor.process(self._make_doc(""))
        assert len(chunks) == 0
