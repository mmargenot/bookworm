import pathlib
import joblib
import yaml

from sklearn.cluster import KMeans
import numpy as np

from bookworm.base import ClusteringMethod
from bookworm.types import Document, Embedding, Cluster


class KMeansClustering(ClusteringMethod):
    """
    Need to be able to reload a fit clusterer
    """

    version = "0.1.0"
    cluster_method = "k-means"

    def __init__(self, k: int):
        self.cluster_model = KMeans(n_clusters=k)
        self.is_fit = False

        self.config = {
            "version": self.version,
            "cluster_method": self.cluster_method,
            "k": k,
            "is_fit": self.is_fit,
        }

    def cluster(
        self, items: list[tuple[Document, Embedding]]
    ) -> list[Cluster]:
        # TODO: clusters of clusters
        # TODO: embeddings have the source id in them already
        # TODO: add dim handling
        documents = []
        embeddings = []

        for d, e in items:
            documents.append(d)
            embeddings.append(e)

        X = np.array([e.embedding for e in embeddings])
        if not self.is_fit:
            self.cluster_model.fit(X)
            self.config["is_fit"] = True
            self.config["dim"] = X.shape[1]
        pred_cluster_labels = self.cluster_model.predict(X)

        clusters = []
        for label_id in np.arange(self.cluster_model.n_clusters):
            doc_idxs = pred_cluster_labels == label_id
            cluster_document_ids = [
                doc.id for (doc, match) in zip(documents, doc_idxs) if match
            ]
            cluster = Cluster(
                label=label_id,
                document_ids=cluster_document_ids,
                parent_id=None,
            )
            clusters.append(cluster)
        return clusters

    @classmethod
    def load(cls, path: str) -> "KMeansClustering":
        _path: pathlib.Path = pathlib.Path(path)
        if not _path.exists():
            raise ValueError(f"Path {path} does not exist")
        if not _path.is_dir():
            raise ValueError(f"Path {path} is not a valid directory")

        config_path = _path / "config.yaml"
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        if not data["version"] == cls.version:
            raise ValueError(
                f"Incorrect version. Expected {cls.version}, but trying to "
                f"load {data['version']}."
            )
        if not data["cluster_method"] == cls.cluster_method:
            raise ValueError(
                f"Cluster method mismatch. Expected {cls.cluster_method}, "
                f"but trying to load {data['cluster_method']}"
            )

        model_path = _path / "model.joblib"
        model = joblib.load(model_path)

        _cls = cls(k=data["k"])
        _cls.cluster_model = model
        _cls.is_fit = data["is_fit"]
        # TODO: log

        return _cls

    def save(self, path: str) -> None:
        _path = pathlib.Path(path)
        _path.mkdir(parents=True, exist_ok=True)
        config_path = _path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config, f, sort_keys=False)

        model_path = _path / "model.joblib"
        joblib.dump(self.cluster_model, model_path)
        # TODO: log
