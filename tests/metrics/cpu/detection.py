import unittest
import numpy as np

from cv_utils.box_utils import calc_boxes_areas
from cv_utils.metrics.cpu.detection import _compute_boxes_iou

__all__ = ['CPUTest']


class CPUTest(unittest.TestCase):
    def test_boxex_iou(self):
        pred = np.array([[[1, 1, 2, 2], [3, 3, 5, 5]], [[1, 3, 2, 5], [-1, -3, 2, 5]]])
        target = pred - np.array([1, 1, 1, 1])
        pred_areas = calc_boxes_areas(pred)
        target_areas = calc_boxes_areas(target)
        res = _compute_boxes_iou(pred[0][0], target, pred_areas[0][0], target_areas)
        self.assertTrue(np.allclose(res, [[0.25 / 1.75, 1 / 3], [0, 2]]))  # TODO: calc last iou

    def test_f_beta(self):
        preds = np.array([[[1, 1, 2, 2], [3, 3, 5, 5]], [[1, 3, 2, 5]]])
        target = np.array([[[1, 1, 2, 2], [3, 3, 5, 5]], [[1, 3, 2, 5]]]) - np.array([1, 1])
