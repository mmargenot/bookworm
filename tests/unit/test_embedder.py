import pytest

from bookworm.embedding import BGEM3Embedder


@pytest.fixture(scope="module")
def embedder():
    return BGEM3Embedder()


class TestBGEM3Embedder:
    def test_empty_list_raises(self, embedder):
        with pytest.raises(ValueError, match="Must submit texts"):
            embedder.embed([])

    def test_empty_string_raises(self, embedder):
        with pytest.raises(ValueError, match="must not contain empty strings"):
            embedder.embed(["hello", ""])

    def test_single_text_returns_one_embedding(self, embedder):
        result = embedder.embed(["hello world"])
        assert len(result) == 1

    def test_multiple_texts_returns_matching_count(self, embedder):
        texts = ["hello", "world", "foo"]
        result = embedder.embed(texts)
        assert len(result) == len(texts)

    def test_embedding_dimensions_consistent(self, embedder):
        result = embedder.embed(["hello", "world"])
        assert len(result[0]) == len(result[1])
