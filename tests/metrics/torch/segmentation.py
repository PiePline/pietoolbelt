import unittest
import torch

from cv_utils.metrics.torch.segmentation import jaccard, dice, multiclass_jaccard, multiclass_dice
from cv_utils.metrics.common import jaccard2dice, dice2jaccard


__all__ = ['PyTorchTest']


class PyTorchTest(unittest.TestCase):
    def test_jaccard(self):
        pred, target = torch.ones((1, 10, 10)), torch.ones((1, 10, 10))
        val = jaccard(pred, target)
        self.assertIsInstance(val, torch.Tensor)
        self.assertTrue(torch.allclose(val, torch.tensor([1], dtype=torch.float32)))
        self.assertEqual(pred.size(), torch.Size([1, 10, 10]))
        self.assertEqual(target.size(), torch.Size([1, 10, 10]))

        pred, target = torch.ones((2, 10, 10)), torch.ones((2, 10, 10))
        pred[0, :, :] = 0
        val = jaccard(pred, target)
        self.assertIsInstance(val, torch.Tensor)
        self.assertTrue(torch.allclose(val, torch.tensor([0, 1], dtype=torch.float32)))
        self.assertEqual(pred.size(), torch.Size([2, 10, 10]))
        self.assertEqual(target.size(), torch.Size([2, 10, 10]))

        pred, target = torch.ones((2, 3, 10, 10)), torch.ones((2, 3, 10, 10))
        pred[0, 0, :5, :] = 0
        target[0, 0, :, :5] = 0
        val = multiclass_jaccard(pred, target)
        self.assertIsInstance(val, torch.Tensor)
        self.assertTrue(torch.allclose(val, torch.tensor([[1 / 3, 1], [1, 1], [1, 1]], dtype=torch.float32)))
        self.assertEqual(pred.size(), torch.Size([2, 3, 10, 10]))
        self.assertEqual(target.size(), torch.Size([2, 3, 10, 10]))

        pred, target = torch.zeros((1, 10, 10)), torch.zeros((1, 10, 10))
        self.assertTrue(torch.allclose(jaccard(pred, target), torch.tensor([1], dtype=torch.float32)))

        pred[0, :5, :] = 1
        target[0, :, :5] = 1
        val = jaccard(pred, target)
        self.assertTrue(torch.allclose(val, torch.tensor([1 / 3], dtype=torch.float32)))

    def test_dice(self):
        pred, target = torch.ones((1, 10, 10)), torch.ones((1, 10, 10))
        val = dice(pred, target)
        self.assertIsInstance(val, torch.Tensor)
        self.assertTrue(torch.allclose(val, torch.tensor([1], dtype=torch.float32)))
        self.assertEqual(pred.size(), torch.Size([1, 10, 10]))
        self.assertEqual(target.size(), torch.Size([1, 10, 10]))

        pred, target = torch.ones((2, 10, 10)), torch.ones((2, 10, 10))
        pred[0, :, :] = 0
        val = dice(pred, target)
        self.assertIsInstance(val, torch.Tensor)
        self.assertTrue(torch.allclose(val, torch.tensor([0, 1], dtype=torch.float32)))
        self.assertEqual(pred.size(), torch.Size([2, 10, 10]))
        self.assertEqual(target.size(), torch.Size([2, 10, 10]))

        pred, target = torch.ones((2, 3, 10, 10)), torch.ones((2, 3, 10, 10))
        pred[0, 0, :5, :] = 0
        target[0, 0, :, :5] = 0
        val = multiclass_dice(pred, target)
        self.assertIsInstance(val, torch.Tensor)
        self.assertTrue(torch.allclose(val, torch.tensor([[0.5, 1], [1, 1], [1, 1]], dtype=torch.float32)))
        self.assertEqual(pred.size(), torch.Size([2, 3, 10, 10]))
        self.assertEqual(target.size(), torch.Size([2, 3, 10, 10]))

        pred, target = torch.zeros((1, 10, 10)), torch.zeros((1, 10, 10))
        self.assertTrue(torch.allclose(dice(pred, target), torch.tensor([1], dtype=torch.float32)))

        pred[0, :5, :] = 1
        target[0, :, :5] = 1
        val = dice(pred, target)
        self.assertTrue(torch.allclose(val, torch.tensor([0.5], dtype=torch.float32)))

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
