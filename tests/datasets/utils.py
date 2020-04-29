import unittest

from cv_utils.datasets.utils import DatasetsContainer
from cv_utils.datasets.common import BasicDataset

__all__ = ['DatasetsContainerTest']


class SimpleDataset(BasicDataset):
    def __init__(self):
        items = list(range(10))
        super().__init__(items)

    def _interpret_item(self, item) -> any:
        return item


class DatasetsContainerTest(unittest.TestCase):
    def test_initialisation(self):
        dataset = DatasetsContainer([SimpleDataset(), SimpleDataset()])
