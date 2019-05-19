import unittest
import numpy as np

from cv_utils.box_utils import calc_boxes_areas
from cv_utils.metrics.cpu.detection import _compute_boxes_iou, f_beta_score

__all__ = ['CPUTest']


class CPUTest(unittest.TestCase):
    def test_boxex_iou(self):
        pred = np.array([[0, 1, 3, 3]])
        target = np.array([[1, 1, 2, 2], [3, 3, 5, 5], [1, 3, 2, 5], [-1, -3, 2, 5]])
        pred_areas = calc_boxes_areas(pred)
        target_areas = calc_boxes_areas(target)
        res = _compute_boxes_iou(pred[0], target, pred_areas[0], target_areas)
        self.assertTrue(np.allclose(res, [1 / 6, 0, 0, 4 / 26]))

    def test_f_beta(self):
        preds = np.array([[[1, 1, 2, 2], [3, 3, 5, 5]], [[1, 3, 2, 5], [-1, -3, 2, 5]]])
        target = np.array([[[1, 1, 2, 2], [3, 3, 5, 5]], [[1, 3, 2, 5], [-1, -3, 2, 5]]]) - np.array([1, 1, 1, 1])

        res = f_beta_score(preds, target, beta=2, thresholds=[0.5, 0.9])
