import unittest

import torch

from cv_utils.models.encoders.mobile_net import MobileNetV2Encoder

__all__ = ['MobileNetV2EncoderTest']


class MobileNetV2EncoderTest(unittest.TestCase):
    def test_init(self):
        try:
            encoder = MobileNetV2Encoder()
        except:
            self.fail("Can't init MobileNetV2 encoder")

    def test_pass_data(self):
        encoder = MobileNetV2Encoder()

        try:
            res = encoder(torch.rand((1, 3, 224,  224)))
        except:
            self.fail("Encoder doesn't pass correct data")

        with self.assertRaises(RuntimeError):
            encoder(torch.rand((1, 2, 224,  224)))

        encoder = MobileNetV2Encoder(input_channels=5, width_mult=2)
        try:
            res = encoder(torch.rand((1, 5, 64,  64)))
            print('a')
        except:
            self.fail("Encoder doesn't pass correct data")
