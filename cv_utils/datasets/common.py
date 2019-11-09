import os
import numpy as np
from abc import ABCMeta, abstractmethod

from neural_pipeline import AbstractDataset


def get_root_by_env(env_name: str) -> str:
    if env_name not in os.environ:
        raise Exception("Can't get dataset root. Please define '" + env_name + "' environment variable")
    return os.environ[env_name]


class BasicDataset(AbstractDataset, metaclass=ABCMeta):
    """
    The standard dataset basic class.

    Basic dataset get array of items and works with it. Array of items is just an array of shape [N, ?]
    """
    def __init__(self, items):
        self._items = items
        self._indices = None

    @abstractmethod
    def _interpret_item(self, item) -> any:
        """
        Interpret one item from dataset. This method get index of item and returns interpreted data? that will be passed from dataset

        Args:
            item: item of items array

        Returns:
            One item, that
        """

    def get_items(self) -> []:
        """
        Get array of items

        :return: array of indices
        """
        return self._items

    def set_indices(self, indices: [int], remove_unused: bool = False) -> 'BasicDataset':
        if remove_unused:
            self._items = [self._items[idx] for idx in indices]
            self._indices = None
        else:
            self._indices = indices
        return self

    def get_indices(self) -> []:
        return self._indices

    def load_indices(self, path: str, remove_unused: bool = False) -> 'BasicDataset':
        self.set_indices(np.load(path), remove_unused)
        return self

    def flush_indices(self, path: str) -> 'BasicDataset':
        np.save(path, self._indices)
        return self

    def __getitem__(self, idx):
        if self._indices is None:
            return self._interpret_item(self._items[idx])
        else:
            return self._interpret_item(self._items[self._indices[idx]])

    def __len__(self):
        return len(self._items)
