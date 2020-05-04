import unittest
import torch
from torch import Tensor
import numpy as np

from pietoolbelt.metrics.torch.segmentation import jaccard, dice, multiclass_jaccard, multiclass_dice, DiceMetric, MulticlassDiceMetric, \
    JaccardMetric, MulticlassJaccardMetric
from pietoolbelt.metrics.common import jaccard2dice, dice2jaccard


__all__ = ['PyTorchTest']


class PyTorchTest(unittest.TestCase):
    def test_jaccard(self):
        self._test_metric(jaccard, multiclass_jaccard, 1 / 3, self._test_functional_metric)

    def test_dice(self):
        self._test_metric(dice, multiclass_dice, 0.5, self._test_functional_metric)

    def test_jaccard_dice_converting(self):
        pred, target = torch.ones((1, 10, 10)), torch.ones((1, 10, 10))
        self.assertEqual(dice(pred, target), jaccard2dice(jaccard(pred, target)))
        self.assertEqual(jaccard(pred, target), dice2jaccard(dice(pred, target)))

        pred, target = torch.zeros((1, 10, 10)), torch.zeros((1, 10, 10))
        self.assertEqual(dice(pred, target), jaccard2dice(jaccard(pred, target)))
        self.assertEqual(jaccard(pred, target), dice2jaccard(dice(pred, target)))

        pred, target = torch.ones((2, 10, 10)), torch.ones((2, 10, 10))
        pred[0, :, :] = 0
        self.assertTrue(torch.allclose(dice(pred, target), jaccard2dice(jaccard(pred, target))))
        self.assertTrue(torch.allclose(jaccard(pred, target), dice2jaccard(dice(pred, target))))

        pred, target = torch.ones((2, 3, 10, 10)), torch.ones((2, 3, 10, 10))
        self.assertTrue(torch.allclose(multiclass_dice(pred, target), jaccard2dice(multiclass_jaccard(pred, target))))
        self.assertTrue(torch.allclose(multiclass_jaccard(pred, target), dice2jaccard(multiclass_dice(pred, target))))

        pred, target = torch.zeros((2, 3, 10, 10)), torch.zeros((2, 3, 10, 10))
        self.assertTrue(torch.allclose(multiclass_dice(pred, target), jaccard2dice(multiclass_jaccard(pred, target))))
        self.assertTrue(torch.allclose(multiclass_jaccard(pred, target), dice2jaccard(multiclass_dice(pred, target))))

    def test_dice_metric(self):
        self._test_metric(DiceMetric().calc, MulticlassDiceMetric(reduction=None).calc, 0.5, self._test_class_metric,
                          multiclass_rediction=None)
        self._test_metric(DiceMetric().calc, MulticlassDiceMetric(reduction='sum').calc, 0.5, self._test_class_metric,
                          multiclass_rediction='sum')
        self._test_metric(DiceMetric().calc, MulticlassDiceMetric(reduction='mean').calc, 0.5, self._test_class_metric,
                          multiclass_rediction='mean')

        self._test_metric(JaccardMetric().calc, MulticlassJaccardMetric(reduction=None).calc, 1 / 3, self._test_class_metric,
                          multiclass_rediction=None)
        self._test_metric(JaccardMetric().calc, MulticlassJaccardMetric(reduction='sum').calc, 1 / 3, self._test_class_metric,
                          multiclass_rediction='sum')
        self._test_metric(JaccardMetric().calc, MulticlassJaccardMetric(reduction='mean').calc, 1 / 3, self._test_class_metric,
                          multiclass_rediction='mean')

    def _test_functional_metric(self, func: callable, pred: Tensor, target: Tensor, expected_res: Tensor):
        pred_size, target_size = pred.size(), target.size()
        val = func(pred, target)
        self.assertIsInstance(val, torch.Tensor)
        self.assertTrue(torch.allclose(val, expected_res))
        self.assertEqual(pred.size(), pred_size)
        self.assertEqual(target.size(), target_size)

    def _test_class_metric(self, func: callable, pred: Tensor, target: Tensor, expected_res: Tensor):
        pred_size, target_size = pred.shape, target.shape
        val = func(pred, target)
        self.assertTrue(type(val) in [float, np.ndarray])
        self.assertTrue(np.allclose(val, expected_res))
        self.assertEqual(pred.shape, pred_size)
        self.assertEqual(target.shape, target_size)

    def _test_metric(self, func: callable, multiclass_func: callable, res_for_half: float, test_method: callable, multiclass_rediction: str = None):
        test_method(func, torch.ones((1, 10, 10)), torch.ones((1, 10, 10)), torch.FloatTensor([1]))

        pred, target = torch.ones((2, 10, 10)), torch.ones((2, 10, 10))
        pred[0, :, :] = 0
        test_method(func, pred, target, torch.FloatTensor([0, 1]))

        pred, target = torch.ones((2, 3, 10, 10)), torch.ones((2, 3, 10, 10))
        pred[0, 0, :5, :] = 0
        target[0, 0, :, :5] = 0
        if multiclass_rediction is None:
            expected_res = torch.FloatTensor([[res_for_half, 1], [1, 1], [1, 1]])
        elif multiclass_rediction == 'sum':
            expected_res = torch.FloatTensor([res_for_half, 1]) + torch.FloatTensor([1, 1]) + torch.FloatTensor([1, 1])
        elif multiclass_rediction == 'mean':
            expected_res = torch.FloatTensor([[res_for_half, 1], [1, 1], [1, 1]]).mean(0)
        test_method(multiclass_func, pred, target, expected_res)

        pred, target = torch.zeros((1, 10, 10)), torch.zeros((1, 10, 10))
        test_method(func, torch.zeros((1, 10, 10)), torch.zeros((1, 10, 10)), torch.FloatTensor([1]))

        pred[0, :5, :] = 1
        target[0, :, :5] = 1
        test_method(func, pred, target, torch.FloatTensor([res_for_half]))
