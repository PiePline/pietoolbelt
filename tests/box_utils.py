import unittest
import numpy as np

from cv_utils.box_utils import calc_boxes_areas

__all__ = ['BoxUtilstest']


class BoxUtilstest(unittest.TestCase):
    def test_calc_boxes_areas(self):
        boxes = np.array([[[1, 1, 2, 2], [3, 3, 5, 5]], [[1, 3, 2, 5], [-1, -3, 2, 5]]])
        res = calc_boxes_areas(boxes)
        self.assertIsInstance(res, np.ndarray)
        self.assertTrue(np.allclose(res, [[1, 4], [2, 24]]))
