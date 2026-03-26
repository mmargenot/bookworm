import pathlib
from abc import abstractmethod
from typing import cast

import joblib
import yaml
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier

from bookworm.base import ClusteringMethod
from bookworm.types import Embedding, Cluster


class SKLearnClusteringMethod(ClusteringMethod):
    """Base class for scikit-learn clustering methods. Holds config and
    serialization logic.

    Args:
        version (str): Semantic version for instance of clustering method,
            used to handle conflicts and changes.
        cluster_method (str): Name of the clustering method.
        cluster_model (BaseEstimator): scikit-learn clustering model.
        is_fit (bool): Flag for whether we need to fit the clusterer to input
            data or not.
    """

    version: str
    cluster_method: str
    cluster_model: BaseEstimator
    is_fit: bool

    @abstractmethod
    def _params(self) -> dict:
        pass

    @property
    def config(self) -> dict:
        return {
            "version": self.version,
            "cluster_method": self.cluster_method,
            "is_fit": self.is_fit,
            **self._params(),
        }

    @classmethod
    def load(cls, path: str) -> "SKLearnClusteringMethod":
        _path = pathlib.Path(path)
        if not _path.exists():
            raise ValueError(f"Path {path} does not exist")
        if not _path.is_dir():
            raise ValueError(f"Path {path} is not a valid directory")

        config_path = _path / "config.yaml"
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        if data["version"] != cls.version:
            raise ValueError(
                f"Incorrect version. Expected {cls.version},"
                f" but trying to load {data['version']}."
            )
        if data["cluster_method"] != cls.cluster_method:
            raise ValueError(
                f"Cluster method mismatch. Expected"
                f" {cls.cluster_method}, but trying to"
                f" load {data['cluster_method']}"
            )

        params = {
            k: v
            for k, v in data.items()
            if k not in ("version", "cluster_method", "is_fit")
        }
        instance = cls(**params)
        instance.cluster_model = joblib.load(_path / "model.joblib")
        instance.is_fit = data["is_fit"]
        return instance

    def save(self, path: str) -> None:
        _path = pathlib.Path(path)
        _path.mkdir(parents=True, exist_ok=True)
        with open(_path / "config.yaml", "w") as f:
            yaml.dump(self.config, f, sort_keys=False)
        joblib.dump(self.cluster_model, _path / "model.joblib")


class KMeansClustering(SKLearnClusteringMethod):
    """Performs K-Means clustering of input embeddings.

    Args:
        k (int): Number of centroids to fit for clustering.
    """

    version = "0.1.0"
    cluster_method = "k-means"
    cluster_model: KMeans

    def __init__(self, k: int):
        self.k = k
        self.cluster_model: KMeans = KMeans(n_clusters=k)
        self.is_fit = False

    def _params(self) -> dict:
        return {"k": self.k}

    def cluster(self, items: list[Embedding]) -> list[Cluster]:
        """Perform clustering on input items."""
        X = np.array([e.embedding for e in items])
        if not self.is_fit:
            self.cluster_model.fit(X)
            self.is_fit = True
        pred_cluster_labels = self.cluster_model.predict(X)

        clusters = []
        for label_id in np.unique(pred_cluster_labels):
            doc_idxs = pred_cluster_labels == label_id
            cluster_document_ids = [
                e.source_id for (e, match) in zip(items, doc_idxs) if match
            ]
            cluster = Cluster(
                label=label_id,
                document_ids=cluster_document_ids,
                parent_id=None,
            )
            clusters.append(cluster)
        return clusters


class DBSCANClustering(SKLearnClusteringMethod):
    """Performs DBSCAN clustering on input embeddings. This is an unsupervised
    density-based clustering algorithm, so it's possible that a sample gets
    mapped into a null "noise" cluster.

    As part of the implementation we include a K-Nearest Neighbors classifier
    that classifies new examples based on old clusters, for continuity.

    Args:
        eps (float): Epsilon distance for grouping min_samples together.
        min_samples (int): Minimum number of samples within eps to make up a
            cluster. Used for the K in the nearest neighbors classifier as
            well.
        metric (str): Metric to use for determining the distance between
            samples. Defaults to `cosine`.
    """

    version = "0.1.0"
    cluster_method = "dbscan"
    cluster_model: DBSCAN

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "cosine",
    ) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_model: DBSCAN = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
        )
        self.nn_classifier = KNeighborsClassifier(
            n_neighbors=min_samples, metric=self.metric
        )
        self.is_fit = False

    def _params(self) -> dict:
        return {
            "eps": self.eps,
            "min_samples": self.min_samples,
            "metric": self.metric,
        }

    def save(self, path: str) -> None:
        super().save(path)
        _path = pathlib.Path(path)
        joblib.dump(
            self.nn_classifier,
            _path / "nn_classifier.joblib",
        )

    @classmethod
    def load(cls, path: str) -> "DBSCANClustering":
        _cls = cast(DBSCANClustering, super().load(path))
        _path = pathlib.Path(path)
        nn_path = _path / "nn_classifier.joblib"
        if nn_path.exists():
            _cls.nn_classifier = joblib.load(nn_path)
        return _cls

    def cluster(self, items: list[Embedding]) -> list[Cluster]:
        X = np.array([e.embedding for e in items])
        if not self.is_fit:
            labels = self.cluster_model.fit_predict(X)
            self.nn_classifier.fit(X, labels)
            self.is_fit = True
        else:
            labels = self.nn_classifier.predict(X)

        clusters = []
        for label_id in np.unique(labels):
            doc_idxs = labels == label_id
            cluster_document_ids = [
                e.source_id for (e, match) in zip(items, doc_idxs) if match
            ]
            cluster = Cluster(
                label=label_id,
                document_ids=cluster_document_ids,
                parent_id=None,
            )
            clusters.append(cluster)
        return clusters
