import unittest
import torch
from torch import Tensor

from pietoolbelt.losses.common import Reduction
from pietoolbelt.losses.segmentation import DiceLoss, MulticlassSegmentationLoss
from pietoolbelt.metrics.torch.segmentation import dice, multiclass_dice

__all__ = ['SegmentationLossesTest']


class SegmentationLossesTest(unittest.TestCase):
    def test_dice(self):
        for batch_reduction in ['sum', 'mean']:
            for multiclass_reduction in ['sum', 'mean']:
                for eps in [1, 1e-7]:
                    with self.subTest(batch_reduction=batch_reduction, multiclass_reduction=multiclass_reduction, eps=eps):
                        self._test_loss(DiceLoss(eps=eps, reduction=Reduction(batch_reduction)),
                                        MulticlassSegmentationLoss(DiceLoss(eps=eps, reduction=Reduction(batch_reduction)),
                                                                   reduction=Reduction(multiclass_reduction)),
                                        batch_reduction=batch_reduction, multiclass_rediction=multiclass_reduction, eps=eps)

    def _test_loss_case(self, func: callable, pred: Tensor, target: Tensor, expected_res: Tensor):
        pred_size, target_size = pred.size(), target.size()
        val = func(pred, target)
        self.assertIsInstance(val, Tensor)
        self.assertEqual(val.size(), torch.Size([]))
        self.assertTrue(torch.allclose(val, expected_res))
        self.assertEqual(pred.size(), pred_size)
        self.assertEqual(target.size(), target_size)

    def _test_loss(self, func: callable, multiclass_func: callable, batch_reduction: str, multiclass_rediction: str, eps: float):
        self._test_loss_case(func, torch.ones((1, 10, 10)), torch.ones((1, 10, 10)), torch.FloatTensor([0]))

        pred, target = torch.ones((2, 10, 10)), torch.ones((2, 10, 10))
        pred[0, :, :] = 0
        expected_res = 1 - dice(pred, target, eps=eps)
        if batch_reduction == 'sum':
            expected_res = expected_res.sum()
        elif batch_reduction == 'mean':
            expected_res = expected_res.mean()
        self._test_loss_case(func, pred, target, expected_res)

        pred, target = torch.ones((2, 3, 10, 10)), torch.ones((2, 3, 10, 10))
        pred[0, 0, :5, :] = 0
        target[0, 0, :, :5] = 0
        expected_res = 1 - multiclass_dice(pred, target, eps=eps)
        if batch_reduction == 'sum':
            expected_res = expected_res.sum(1)
        elif batch_reduction == 'mean':
            expected_res = expected_res.mean(1)
        if multiclass_rediction == 'sum':
            expected_res = expected_res.sum(0)
        elif multiclass_rediction == 'mean':
            expected_res = expected_res.mean(0)
        self._test_loss_case(multiclass_func, pred, target, expected_res)

        pred, target = torch.zeros((1, 10, 10)), torch.zeros((1, 10, 10))
        self._test_loss_case(func, torch.zeros((1, 10, 10)), torch.zeros((1, 10, 10)), 1 - dice(pred, target, eps=eps))

        pred[0, :5, :] = 1
        target[0, :, :5] = 1
        self._test_loss_case(func, pred, target, 1 - dice(pred, target, eps=eps))
