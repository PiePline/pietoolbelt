import unittest
import numpy as np

from pietoolbelt.tiles_utils import TilesManager


class BasicEncoderTest(unittest.TestCase):
    def test_init(self):
        try:
            TilesManager()
        except Exception as err:
            self.fail(err)

    def test_tiles_generation_without_overlap(self):
        tiles = TilesManager().generate_tiles([[0, 0], [512, 512]], [256, 256]).get_tiles()
        self.assertTrue(np.array_equal(tiles, [[[0, 0], [256, 256]], [[256, 0], [512, 256]], [[0, 256], [256, 512]], [[256, 256], [512, 512]]]))

        tiles = TilesManager().generate_tiles([[0, 0], [1024, 512]], [512, 512]).get_tiles()
        self.assertTrue(np.array_equal(tiles, [[[0, 0], [512, 512]], [[512, 0], [1024, 512]]]))

        tiles = TilesManager().generate_tiles([[0, 0], [512, 1024]], [512, 512]).get_tiles()
        self.assertTrue(np.array_equal(tiles, [[[0, 0], [512, 512]], [[0, 512], [512, 1024]]]))

    def test_tiles_generation_with_overlap(self):
        tiles = TilesManager().generate_tiles([[0, 0], [1024, 512]], [512 + 256, 512]).get_tiles()
        self.assertTrue(np.array_equal(tiles, [[[0, 0], [512 + 256 // 2, 512]], [[512 - 256 // 2, 0], [1024, 512]]]))
