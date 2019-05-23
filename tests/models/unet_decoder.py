import os
import unittest

import torch

from cv_utils.models.decoders.unet import UNetDecoder
from cv_utils.models.encoders.inception import InceptionV3Encoder
from cv_utils.models.encoders.mobile_net import MobileNetV2Encoder
from cv_utils.models.encoders.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

__all__ = ['ResNet18EncoderTest']


class BasicUnetTest(unittest.TestCase):
    def _init_unet(self, in_channels: int):
        return None

    _min_img_size = None
    _name = None
    _layers_num = None
    _get_out_size_by_input_size = None

    def test_init(self):
        try:
            encoder = self._init_unet(3)
        except:
            self.fail("Can't init {} encoder".format(self._name))

    def test_pass_data(self):
        for in_channels in [1, 3]:
            for batch_size in [1, 3]:
                for img_size in [self._min_img_size + 64 * i for i in range(3)]:
                    with self.subTest(in_channels=in_channels, batch_size=batch_size, img_size=img_size):
                        try:
                            encoder = self._init_unet(in_channels=in_channels)
                            res = encoder(torch.rand((batch_size, in_channels, img_size, img_size)))
                            self.assertEqual(res.size(), self._get_out_size_by_input_size(batch_size, in_channels, img_size, img_size))
                            if img_size // 2 >= self._min_img_size:
                                res = encoder(torch.rand((batch_size, in_channels, img_size // 2, img_size)))
                                self.assertEqual(res.size(), self._get_out_size_by_input_size(batch_size, in_channels, img_size // 2, img_size))
                                res = encoder(torch.rand((batch_size, in_channels, img_size, img_size // 2)))
                                self.assertEqual(res.size(), self._get_out_size_by_input_size(batch_size, in_channels, img_size, img_size // 2))
                        except:
                            self.fail("Encoder doesn't pass correct data")

    def test_jit(self):
        encoder = self._init_unet(in_channels=2).eval()

        try:
            torch.jit.trace(encoder, torch.rand(1, 2, self._min_img_size, self._min_img_size))
        except:
            self.fail("Fail to trace model")

    # def test_onnx(self):
    #     encoder = self._init_unet(in_channels=2).eval()
    #
    #     try:
    #         torch.onnx.export(encoder, torch.rand(1, 2, self._min_img_size, self._min_img_size), 'test_onnx.onnx', verbose=False)
    #         self.assertTrue(os.path.exists('test_onnx.onnx') and os.path.isfile('test_onnx.onnx'))
    #     except:
    #         self.fail("Fail to trace model")
    #     os.remove('test_onnx.onnx')


class ResNet18EncoderTest(BasicUnetTest):
    _min_img_size = 64
    _name = "ResNet18Encoder"
    _layers_num = 5

    def _init_unet(self, in_channels: int):
        return UNetDecoder(resnet18(in_channels), classes_num=1)

    @staticmethod
    def _get_out_size_by_input_size(batch_size, in_channels, img_size_x, img_size_y):
        return torch.Size([batch_size, 1, img_size_x, img_size_y])


class ResNet34EncoderTest(BasicUnetTest):
    _min_img_size = 64
    _name = "ResNet18Encoder"
    _layers_num = 5

    def _init_unet(self, in_channels: int):
        return UNetDecoder(resnet34(in_channels), classes_num=1)

    @staticmethod
    def _get_out_size_by_input_size(batch_size, in_channels, img_size_x, img_size_y):
        return torch.Size([batch_size, 1, img_size_x, img_size_y])


class ResNet50EncoderTest(BasicUnetTest):
    _min_img_size = 64
    _name = "ResNet18Encoder"
    _layers_num = 5

    def _init_unet(self, in_channels: int):
        return UNetDecoder(resnet50(in_channels), classes_num=1)

    @staticmethod
    def _get_out_size_by_input_size(batch_size, in_channels, img_size_x, img_size_y):
        return torch.Size([batch_size, 1, img_size_x, img_size_y])