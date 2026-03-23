import pytest

from bookworm.types import Document, Embedding, EmbeddingType, Source
from bookworm.embedding import BGEM3Embedder
from bookworm.cluster_method import KMeansClustering


JABBERWOCKY = [
    "'Twas brillig, and the slithy toves did gyre and gimble in the wabe; "
    "all mimsy were the borogoves, and the mome raths outgrabe.",
    "Beware the Jabberwock, my son! The jaws that bite, the claws that catch! "
    "Beware the Jubjub bird, and shun the frumious Bandersnatch!",
    "He took his vorpal sword in hand; long time the manxome foe he sought— "
    "so rested he by the Tumtum tree, and stood awhile in thought.",
    "And, as in uffish thought he stood, the Jabberwock, with eyes of flame, "
    "came whiffling through the tulgey wood, and burbled as it came!",
    "One, two! One, two! And through and through the vorpal blade went snicker-snack! "
    "He left it dead, and with its head he went galumphing back.",
    "'And hast thou slain the Jabberwock? Come to my arms, my beamish boy! "
    "O frabjous day! Callooh! Callay!' He chortled in his joy.",
    "'Twas brillig, and the slithy toves did gyre and gimble in the wabe; "
    "all mimsy were the borogoves, and the mome raths outgrabe.",
]


@pytest.fixture(scope="module")
def embedder():
    return BGEM3Embedder()


@pytest.fixture(scope="module")
def documents() -> list[Document]:
    return [
        Document(name=f"stanza_{i}", source=Source(), content=text)
        for i, text in enumerate(JABBERWOCKY)
    ]


@pytest.fixture(scope="module")
def embedded_documents(embedder, documents):
    raw_embeddings = embedder.embed([doc.content for doc in documents])
    items = []
    for doc, emb in zip(documents, raw_embeddings):
        embedding = Embedding(
            source_id=doc.id,
            type=EmbeddingType.TEXT,
            embedding=emb,
        )
        items.append((doc, embedding))
    return items


class TestEmbedThenCluster:
    def test_cluster_count_matches_k(self, embedded_documents):
        k = 3
        clusterer = KMeansClustering(k=k)
        clusters = clusterer.cluster(embedded_documents)
        assert len(clusters) == k

    def test_all_documents_assigned(self, embedded_documents):
        k = 3
        clusterer = KMeansClustering(k=k)
        clusters = clusterer.cluster(embedded_documents)
        all_doc_ids = {doc.id for doc, _ in embedded_documents}
        clustered_doc_ids = {
            doc_id for cluster in clusters for doc_id in cluster.document_ids
        }
        assert clustered_doc_ids == all_doc_ids

    def test_no_empty_clusters(self, embedded_documents):
        k = 3
        clusterer = KMeansClustering(k=k)
        clusters = clusterer.cluster(embedded_documents)
        for cluster in clusters:
            assert len(cluster.document_ids) > 0
