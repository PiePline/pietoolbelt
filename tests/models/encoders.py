import os
import unittest

import torch

from pietoolbelt.models.encoders.inception import InceptionV3Encoder
from pietoolbelt.models.encoders.mobile_net import MobileNetV2Encoder
from pietoolbelt.models.encoders.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

__all__ = ['MobileNetV2EncoderTest', 'InceptionV3EncoderTest', 'ResNet18EncoderTest', 'ResNet34EncoderTest', 'ResNet50EncoderTest', 'ResNet101EncoderTest', 'ResNet152EncoderTest']


class BasicEncoderTest(unittest.TestCase):
    def _init_encoder(self, in_channels: int):
        return None

    _min_img_size = None
    _name = None
    _layers_num = None
    _get_out_size_by_input_size = None

    def test_init(self):
        try:
            encoder = self._init_encoder(3)
        except:
            self.fail("Can't init {} encoder".format(self._name))

    def test_pass_data(self):
        for in_channels in [1, 3]:
            for batch_size in [1, 3]:
                for img_size in [self._min_img_size + 64 * i for i in range(3)]:
                    with self.subTest(in_channels=in_channels, batch_size=batch_size, img_size=img_size):
                        try:
                            encoder = self._init_encoder(in_channels=in_channels)
                            res = encoder(torch.rand((batch_size, in_channels, img_size, img_size)))
                            self.assertEqual(res.size(), self._get_out_size_by_input_size(batch_size, in_channels, img_size, img_size))
                            if img_size // 2 >= self._min_img_size:
                                res = encoder(torch.rand((batch_size, in_channels, img_size // 2, img_size)))
                                self.assertEqual(res.size(), self._get_out_size_by_input_size(batch_size, in_channels, img_size // 2, img_size))
                                res = encoder(torch.rand((batch_size, in_channels, img_size, img_size // 2)))
                                self.assertEqual(res.size(), self._get_out_size_by_input_size(batch_size, in_channels, img_size, img_size // 2))
                        except:
                            self.fail("Encoder doesn't pass correct data")

    def test_layers_outputs_collecting(self):
        encoder = self._init_encoder(in_channels=5)
        encoder(torch.rand((1, 5, self._min_img_size, self._min_img_size)))
        self.assertIsNone(encoder.get_layers_outputs())

        encoder.collect_layers_outputs(True)
        encoder(torch.rand((1, 5, self._min_img_size, self._min_img_size)))
        self.assertEqual(len(encoder.get_layers_outputs()), self._layers_num)

    def test_jit(self):
        encoder = self._init_encoder(in_channels=2).eval()

        try:
            torch.jit.trace(encoder, torch.rand(1, 2, self._min_img_size, self._min_img_size))
        except:
            self.fail("Fail to trace model")

    def test_onnx(self):
        encoder = self._init_encoder(in_channels=2).eval()

        try:
            torch.onnx.export(encoder, torch.rand(1, 2, self._min_img_size, self._min_img_size), 'test_onnx.onnx', verbose=False)
            self.assertTrue(os.path.exists('test_onnx.onnx') and os.path.isfile('test_onnx.onnx'))
        except:
            self.fail("Fail to trace model")
        os.remove('test_onnx.onnx')


class MobileNetV2EncoderTest(BasicEncoderTest):
    # _init_encoder = lambda in_channels: MobileNetV2Encoder(in_channels)
    _min_img_size = 64
    _name = "MobileNetV2Encoder"
    _layers_num = 7

    def _init_encoder(self, in_channels: int):
        return MobileNetV2Encoder(in_channels)

    @staticmethod
    def _get_out_size_by_input_size(batch_size, in_channels, img_size_x, img_size_y):
        return torch.Size([batch_size, 1280, img_size_x // 32, img_size_y // 32])


class InceptionV3EncoderTest(BasicEncoderTest):
    _min_img_size = 128
    _name = "InceptionV3Encoder"
    _layers_num = 11

    def _init_encoder(self, in_channels: int):
        return InceptionV3Encoder(in_channels)

    @staticmethod
    def _get_out_size_by_input_size(batch_size, in_channels, img_size_x, img_size_y):
        return torch.Size([batch_size, 2048, (img_size_x - 64) // 32, (img_size_y - 64) // 32])


class ResNet18EncoderTest(BasicEncoderTest):
    _min_img_size = 64
    _name = "ResNet18Encoder"
    _layers_num = 4

    def _init_encoder(self, in_channels: int):
        return ResNet18(in_channels)

    @staticmethod
    def _get_out_size_by_input_size(batch_size, in_channels, img_size_x, img_size_y):
        return torch.Size([batch_size, 512, img_size_x // 32, img_size_y // 32])


class ResNet34EncoderTest(BasicEncoderTest):
    _min_img_size = 64
    _name = "ResNet18Encoder"
    _layers_num = 4

    def _init_encoder(self, in_channels: int):
        return ResNet34(in_channels)

    @staticmethod
    def _get_out_size_by_input_size(batch_size, in_channels, img_size_x, img_size_y):
        return torch.Size([batch_size, 512, img_size_x // 32, img_size_y // 32])


class ResNet50EncoderTest(BasicEncoderTest):
    _min_img_size = 64
    _name = "ResNet18Encoder"
    _layers_num = 4

    def _init_encoder(self, in_channels: int):
        return ResNet50(in_channels)

    @staticmethod
    def _get_out_size_by_input_size(batch_size, in_channels, img_size_x, img_size_y):
        return torch.Size([batch_size, 2048, img_size_x // 32, img_size_y // 32])


class ResNet101EncoderTest(BasicEncoderTest):
    _min_img_size = 64
    _name = "ResNet18Encoder"
    _layers_num = 4

    def _init_encoder(self, in_channels: int):
        return ResNet101(in_channels)

    @staticmethod
    def _get_out_size_by_input_size(batch_size, in_channels, img_size_x, img_size_y):
        return torch.Size([batch_size, 2048, img_size_x // 32, img_size_y // 32])


class ResNet152EncoderTest(BasicEncoderTest):
    _min_img_size = 64
    _name = "ResNet18Encoder"
    _layers_num = 4

    def _init_encoder(self, in_channels: int):
        return ResNet152(in_channels)

    @staticmethod
    def _get_out_size_by_input_size(batch_size, in_channels, img_size_x, img_size_y):
        return torch.Size([batch_size, 2048, img_size_x // 32, img_size_y // 32])
