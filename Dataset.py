from pathlib import Path
import numpy as np
import scipy.sparse as sp


class Dataset:
    """Utility class for loading and processing rating datasets."""

    def __init__(self, path: str):
        """
        Initialize the dataset.

        Args:
            path (str): Base path (without extension) for dataset files.
                        Expects files:
                        - {path}.train.rating
                        - {path}.test.rating
                        - {path}.test.negative
        """
        base = Path(path)

        self.train_matrix = self._load_rating_file_as_matrix(base.with_suffix(".train.rating"))
        self.test_ratings = self._load_rating_file_as_list(base.with_suffix(".test.rating"))
        self.test_negatives = self._load_negative_file(base.with_suffix(".test.negative"))

        assert len(self.test_ratings) == len(self.test_negatives), \
            "Mismatch between test ratings and negatives"

        self.num_users, self.num_items = self.train_matrix.shape

    @staticmethod
    def _load_rating_file_as_list(filename: Path):
        """Load rating file into a list of [user, item] pairs."""
        rating_list = []
        with filename.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                user, item, *_ = map(int, line.strip().split("\t"))
                rating_list.append([user, item])
        return rating_list

    @staticmethod
    def _load_negative_file(filename: Path):
        """Load negative samples into a list of lists."""
        negative_list = []
        with filename.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                _, *negatives = line.strip().split("\t")
                negative_list.append([int(x) for x in negatives])
        return negative_list

    @staticmethod
    def _load_rating_file_as_matrix(filename: Path):
        """
        Read .rating file and return a DOK matrix.
        The first line of .rating file is: num_users \t num_items
        """
        # Determine max user and item IDs
        num_users, num_items = 0, 0
        with filename.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                u, i, *_ = map(int, line.strip().split("\t")[:2])
                num_users = max(num_users, u)
                num_items = max(num_items, i)

        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with filename.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                user, item, rating = line.strip().split("\t")
                user, item, rating = int(user), int(item), float(rating)
                if rating > 0:
                    mat[user, item] = 1.0
        return mat
