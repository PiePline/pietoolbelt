import os
import unittest

import torch

from cv_utils.models.encoders.inception import InceptionV3Encoder
from cv_utils.models.encoders.mobile_net import MobileNetV2Encoder

__all__ = ['MobileNetV2EncoderTest', 'InceptionV3EncoderTest']


class MobileNetV2EncoderTest(unittest.TestCase):
    def test_init(self):
        try:
            encoder = MobileNetV2Encoder()
        except:
            self.fail("Can't init MobileNetV2 encoder")

    def test_pass_data(self):
        for in_channels in [1, 3]:
            for batch_size in [1, 3]:
                for img_size in [64 + 32 * i for i in range(3)]:
                    with self.subTest(in_channels=in_channels, batch_size=batch_size, img_size=img_size):
                        try:
                            encoder = MobileNetV2Encoder(input_channels=in_channels)
                            encoder(torch.rand((batch_size, in_channels, img_size, img_size)))
                            if img_size > 64 and img_size % 64 == 0:
                                encoder(torch.rand((batch_size, in_channels, img_size // 2, img_size)))
                                encoder(torch.rand((batch_size, in_channels, img_size, img_size // 2)))

                            encoder = MobileNetV2Encoder(input_channels=in_channels, width_mult=2)
                            res = encoder(torch.rand((batch_size, in_channels, img_size, img_size)))
                            self.assertEqual(res.size(), torch.Size([batch_size, 1280 * 2, img_size // 32, img_size // 32]))
                        except:
                            self.fail("Encoder doesn't pass correct data")

        encoder = MobileNetV2Encoder(input_channels=5, width_mult=2)
        with self.assertRaises(RuntimeError):
            encoder(torch.rand((batch_size, in_channels, img_size, img_size)))

    def test_layers_outputs_collecting(self):
        encoder = MobileNetV2Encoder(input_channels=5, width_mult=2)
        encoder(torch.rand((1, 5, 64, 64)))
        self.assertIsNone(encoder.get_layers_outputs())

        encoder.collect_layers_outputs(True)
        encoder(torch.rand((1, 5, 64, 64)))
        self.assertEqual(len(encoder.get_layers_outputs()), 8)

    def test_jit(self):
        encoder = MobileNetV2Encoder(input_channels=2, width_mult=1)

        try:
            torch.jit.trace(encoder, torch.rand(1, 2, 64, 64))
        except:
            self.fail("Fail to trace model")

    def test_onnx(self):
        encoder = MobileNetV2Encoder(input_channels=2, width_mult=1)

        try:
            torch.onnx.export(encoder, torch.rand(1, 2, 64, 64), 'test_onnx.onnx', verbose=False)
            self.assertTrue(os.path.exists('test_onnx.onnx') and os.path.isfile('test_onnx.onnx'))
        except:
            self.fail("Fail to trace model")
        os.remove('test_onnx.onnx')


class InceptionV3EncoderTest(unittest.TestCase):
    def test_init(self):
        try:
            encoder = InceptionV3Encoder()
        except:
            self.fail("Can't init MobileNetV2 encoder")

    def test_pass_data(self):
        for in_channels in [1, 3]:
            for batch_size in [1, 3]:
                for img_size in [128 + 64 * i for i in range(3)]:
                    with self.subTest(in_channels=in_channels, batch_size=batch_size, img_size=img_size):
                        try:
                            encoder = InceptionV3Encoder(input_channels=in_channels)
                            res = encoder(torch.rand((batch_size, in_channels, img_size, img_size)))
                            self.assertEqual(res.size(), torch.Size([batch_size, 2048, 1, 1]))
                            if img_size // 2 >= 128:
                                res = encoder(torch.rand((batch_size, in_channels, img_size // 2, img_size)))
                                self.assertEqual(res.size(), torch.Size([batch_size, 2048, 1, 1]))
                                res = encoder(torch.rand((batch_size, in_channels, img_size, img_size // 2)))
                                self.assertEqual(res.size(), torch.Size([batch_size, 2048, 1, 1]))
                        except:
                            self.fail("Encoder doesn't pass correct data")

    def test_layers_outputs_collecting(self):
        encoder = InceptionV3Encoder(input_channels=5)
        encoder(torch.rand((1, 5, 64 * 2, 64 * 2)))
        self.assertIsNone(encoder.get_layers_outputs())

        encoder.collect_layers_outputs(True)
        encoder(torch.rand((1, 5, 64 * 2, 64 * 2)))
        self.assertEqual(len(encoder.get_layers_outputs()), 12)
