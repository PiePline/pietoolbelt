import unittest
import numpy as np

from pietoolbelt.metrics.cpu.segmentation import jaccard, dice, multiclass_jaccard, multiclass_dice
from pietoolbelt.metrics.common import jaccard2dice, dice2jaccard

__all__ = ['CPUTest']


class CPUTest(unittest.TestCase):
    def test_jaccard(self):
        pred, target = np.ones((1, 10, 10)), np.ones((1, 10, 10))
        val = jaccard(pred, target)
        self.assertIsInstance(val, np.ndarray)
        self.assertTrue(np.allclose(val, [1]))
        self.assertEqual(pred.shape, (1, 10, 10))
        self.assertEqual(target.shape, (1, 10, 10))

        pred, target = np.ones((2, 10, 10)), np.ones((2, 10, 10))
        pred[0, :, :] = 0
        val = jaccard(pred, target)
        self.assertIsInstance(val, np.ndarray)
        self.assertTrue(np.allclose(val, [0, 1]))
        self.assertEqual(pred.shape, (2, 10, 10))
        self.assertEqual(target.shape, (2, 10, 10))

        pred, target = np.ones((2, 3, 10, 10)), np.ones((2, 3, 10, 10))
        pred[0, 0, :, :] = 0
        val = multiclass_jaccard(pred, target)
        self.assertIsInstance(val, np.ndarray)
        self.assertTrue(np.allclose(val, [[0, 1], [1, 1], [1, 1]]))
        self.assertEqual(pred.shape, (2, 3, 10, 10))
        self.assertEqual(target.shape, (2, 3, 10, 10))

        pred, target = np.zeros((1, 10, 10)), np.zeros((1, 10, 10))
        self.assertTrue(np.allclose(jaccard(pred, target), [1]))

        pred[0, :5, :] = 1
        target[0, :, :5] = 1
        val = jaccard(pred, target)
        self.assertTrue(np.allclose(val, [1 / 3]))

    def test_dice(self):
        pred, target = np.ones((1, 10, 10)), np.ones((1, 10, 10))
        val = dice(pred, target)
        self.assertIsInstance(val, np.ndarray)
        self.assertTrue(np.allclose(val, [1]))
        self.assertEqual(pred.shape, (1, 10, 10))
        self.assertEqual(target.shape, (1, 10, 10))

        pred, target = np.ones((2, 10, 10)), np.ones((2, 10, 10))
        pred[0, :, :] = 0
        val = dice(pred, target)
        self.assertIsInstance(val, np.ndarray)
        self.assertTrue(np.allclose(val, [0, 1]))
        self.assertEqual(pred.shape, (2, 10, 10))
        self.assertEqual(target.shape, (2, 10, 10))

        pred, target = np.ones((2, 3, 10, 10)), np.ones((2, 3, 10, 10))
        pred[0, 0, :, :] = 0
        val = multiclass_dice(pred, target)
        self.assertIsInstance(val, np.ndarray)
        self.assertTrue(np.allclose(val, [[0, 1], [1, 1], [1, 1]]))
        self.assertEqual(pred.shape, (2, 3, 10, 10))
        self.assertEqual(target.shape, (2, 3, 10, 10))

        pred, target = np.zeros((1, 10, 10)), np.zeros((1, 10, 10))
        self.assertTrue(np.allclose(dice(pred, target), [1]))

        pred[0, :5, :] = 1
        target[0, :, :5] = 1
        val = dice(pred, target)
        self.assertTrue(np.allclose(val, [0.5]))

    def test_jaccard_dice_converting(self):
        pred, target = np.ones((1, 10, 10)), np.ones((1, 10, 10))
        self.assertEqual(dice(pred, target), jaccard2dice(jaccard(pred, target)))
        self.assertEqual(jaccard(pred, target), dice2jaccard(dice(pred, target)))

        pred, target = np.zeros((1, 10, 10)), np.zeros((1, 10, 10))
        self.assertEqual(dice(pred, target), jaccard2dice(jaccard(pred, target)))
        self.assertEqual(jaccard(pred, target), dice2jaccard(dice(pred, target)))

        pred, target = np.ones((2, 10, 10)), np.ones((2, 10, 10))
        pred[0, :, :] = 0
        self.assertTrue(np.allclose(dice(pred, target), jaccard2dice(jaccard(pred, target))))
        self.assertTrue(np.allclose(jaccard(pred, target), dice2jaccard(dice(pred, target))))

        pred, target = np.ones((2, 3, 10, 10)), np.ones((2, 3, 10, 10))
        self.assertTrue(np.allclose(multiclass_dice(pred, target), jaccard2dice(multiclass_jaccard(pred, target))))
        self.assertTrue(np.allclose(multiclass_jaccard(pred, target), dice2jaccard(multiclass_dice(pred, target))))

        pred, target = np.zeros((2, 3, 10, 10)), np.zeros((2, 3, 10, 10))
        self.assertTrue(np.allclose(multiclass_dice(pred, target), jaccard2dice(multiclass_jaccard(pred, target))))
        self.assertTrue(np.allclose(multiclass_jaccard(pred, target), dice2jaccard(multiclass_dice(pred, target))))
