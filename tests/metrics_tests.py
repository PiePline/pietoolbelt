import unittest
import numpy as np

from cv_utils.metrics.numpy.segmentation import jaccard

__all__ = ['MetricsTest']


class MetricsTest(unittest.TestCase):
    def test_jaccard_numpy(self):
        pred = np.ones((1, 10, 10))
        target = np.ones((1, 10, 10))
        self.assertEqual(jaccard(pred, target), 1)

        pred = np.zeros((1, 10, 10))
        target = np.zeros((1, 10, 10))
        self.assertEqual(jaccard(pred, target), 1)
