import unittest
import torch
from torch import Tensor

from cv_utils.losses.segmentation import DiceLoss, MulticlassSegmentationLoss
from cv_utils.metrics.torch.segmentation import dice, multiclass_dice

__all__ = ['SegmentationLossesTest']


class SegmentationLossesTest(unittest.TestCase):
    def test_dice(self):
        pred, target = torch.ones((1, 10, 10)), torch.ones((1, 10, 10))

        true_val = dice(pred, target, eps=1)
        res = DiceLoss(eps=1, activation=None)(pred, target)
        self.assertTrue(torch.allclose(true_val, 1 - res))

        true_val = dice(pred, target, eps=1e-7)
        res = DiceLoss(eps=1e-7, activation=None)(pred, target)
        self.assertTrue(torch.allclose(true_val, 1 - res))
        self.assertIsInstance(res, torch.Tensor)
        self.assertEqual(res.size(), torch.Size([1]))

        pred[0, :5, :] = 0
        target[0, :, :5] = 0
        val = DiceLoss(eps=1e-7, activation=None)(pred, target)
        self.assertTrue(torch.allclose(val, 1 - torch.tensor([0.5], dtype=torch.float32)))
        self.assertIsInstance(res, torch.Tensor)
        self.assertEqual(res.size(), torch.Size([1]))

    def test_multiclass_loss(self):
        pred, target = torch.ones((2, 10, 10)), torch.ones((2, 10, 10))
        true_val = dice(pred, target, eps=1e-7)
        res = DiceLoss(eps=1e-7, activation=None)(pred, target)
        self.assertTrue(torch.allclose(true_val, 1 - res))

        pred, target = torch.ones((2, 3, 10, 10)), torch.ones((2, 3, 10, 10))
        true_val = multiclass_dice(pred, target, eps=1)
        res = MulticlassSegmentationLoss(DiceLoss(eps=1, activation=None))(pred, target)
        self.assertTrue(torch.allclose(true_val, 1 - res))

    def _test_class_metric(self, func: callable, pred: Tensor, target: Tensor, expected_res: Tensor):
        pred_size, target_size = pred.size(), target.size()
        val = func(pred, target)
        self.assertIsInstance(val, Tensor)
        self.assertTrue(torch.allclose(val, expected_res))
        self.assertEqual(pred.size(), pred_size)
        self.assertEqual(target.size(), target_size)

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
