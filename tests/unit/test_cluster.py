import pytest
import yaml

from bookworm.cluster_method import KMeansClustering


class TestKMeansClusteringLoad:
    def test_load_nonexistent_path(self, tmp_path):
        bad_path = str(tmp_path / "nonexistent")
        with pytest.raises(ValueError, match="does not exist"):
            KMeansClustering.load(bad_path)

    def test_load_file_not_directory(self, tmp_path):
        file_path = tmp_path / "not_a_dir.txt"
        file_path.touch()
        with pytest.raises(ValueError, match="is not a valid directory"):
            KMeansClustering.load(str(file_path))

    def test_load_version_mismatch(self, tmp_path):
        config = {
            "version": "999.0.0",
            "cluster_method": "k-means",
            "k": 3,
            "is_fit": False,
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with pytest.raises(ValueError, match="Incorrect version"):
            KMeansClustering.load(str(tmp_path))

    def test_load_cluster_method_mismatch(self, tmp_path):
        config = {
            "version": KMeansClustering.version,
            "cluster_method": "dbscan",
            "k": 3,
            "is_fit": False,
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with pytest.raises(ValueError, match="Cluster method mismatch"):
            KMeansClustering.load(str(tmp_path))
