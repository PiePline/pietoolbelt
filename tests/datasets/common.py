import os
import unittest

from pietoolbelt.datasets.common import BasicDataset

__all__ = ['BasicDatasetTest']


class SimpleDataset(BasicDataset):
    def __init__(self):
        items = list(range(10))
        super().__init__(items)

    def _interpret_item(self, item) -> any:
        return item


class BasicDatasetTest(unittest.TestCase):
    def test_basic_functional(self):
        dataset = SimpleDataset()
        expected_res = list(range(10))
        self.assertEqual(dataset.get_items(), expected_res)

        res = []
        for data in dataset:
            res.append(data)
        self.assertEqual(res, expected_res)

    def test_indices(self):
        dataset = SimpleDataset()
        indices = [1, 2, 3, 7, 9]
        expected_res = list(range(10))

        dataset.set_indices(indices)
        self.assertEqual(dataset.get_items(), expected_res)
        expected_res = [expected_res[i] for i in indices]
        dataset.set_indices(indices).remove_unused_data()
        self.assertEqual(dataset.get_items(), expected_res)

        res = []
        for data in dataset:
            res.append(data)
        self.assertEqual(res, expected_res)

        dataset.set_indices(indices)
        dataset.flush_indices('test_indices.npy')
        del dataset
        dataset = SimpleDataset()

        dataset.load_indices('test_indices.npy')
        expected_res = list(range(10))
        self.assertEqual(dataset.get_items(), expected_res)

        dataset.load_indices('test_indices.npy').remove_unused_data()
        expected_res = [expected_res[i] for i in indices]
        self.assertEqual(dataset.get_items(), expected_res)

        self.assertTrue(os.path.exists('test_indices.npy') and os.path.isfile('test_indices.npy'))
        os.remove('test_indices.npy')
