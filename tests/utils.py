import unittest
import numpy as np

from pietoolbelt.utils import get_from_dict, put_to_dict, mask2rle, rle2mask


class UtilsTest(unittest.TestCase):
    def test_get_from_dict(self):
        real_dict = {'a': {'b': {'c': 4}, 'd': {'dd'}}}
        d = real_dict.copy()

        self.assertEqual(get_from_dict(d, ['a', 'b', 'c']), 4)
        self.assertEqual(get_from_dict(d, ['a', 'd']), {'dd'})

        self.assertEqual(real_dict, d)

    def test_put_to_dict(self):
        d = {'a': {'b': {'c': 4}, 'd': {'dd'}}}

        self.assertEqual(put_to_dict(d, ['a', 'b', 'e'], 8), {'a': {'b': {'c': 4, 'e': 8}, 'd': {'dd'}}})
        self.assertEqual(d, {'a': {'b': {'c': 4, 'e': 8}, 'd': {'dd'}}})

    def test_rle(self):
        mask = np.zeros((512, 256), dtype=np.uint8)
        mask[23: 76, 56: 78] = 255

        rle = mask2rle(mask)
        new_mask = rle2mask([int(v) for v in rle.split(' ')], (512, 256))
        self.assertTrue(np.equal(mask, new_mask))
