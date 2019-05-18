import unittest
import torch

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

        pred[0, :5, :] = 0
        target[0, :, :5] = 0
        val = DiceLoss(eps=1e-7, activation=None)(pred, target)
        self.assertTrue(torch.allclose(val, 1 - torch.tensor([0.5], dtype=torch.float32)))

    def test_multiclass_loss(self):
        pred, target = torch.ones((2, 10, 10)), torch.ones((2, 10, 10))
        true_val = dice(pred, target, eps=1e-7)
        res = DiceLoss(eps=1e-7, activation=None)(pred, target)
        self.assertTrue(torch.allclose(true_val, 1 - res))

        pred, target = torch.ones((2, 3, 10, 10)), torch.ones((2, 3, 10, 10))
        true_val = multiclass_dice(pred, target, eps=1)
        res = MulticlassSegmentationLoss(DiceLoss(eps=1, activation=None))(pred, target)
        self.assertTrue(torch.allclose(true_val, 1 - res))
