import os
import unittest

import torch

from pietoolbelt.models.decoders.unet import UNetDecoder
from pietoolbelt.models.encoders.inception import InceptionV3Encoder
from pietoolbelt.models.encoders.mobile_net import MobileNetV2Encoder
from pietoolbelt.models.encoders.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

__all__ = ['ResNet18DecoderTest', 'ResNet34DecoderTest', 'ResNet50DecoderTest', 'ResNet101DecoderTest', 'ResNet152DecoderTest']


class BasicUnetTest(unittest.TestCase):
    def _init_unet(self, in_channels: int, classes_num: int):
        return None

    _min_img_size = None
    _name = None
    _layers_num = None
    _get_out_size_by_input_size = None

    def test_init(self):
        try:
            encoder = self._init_unet(3, 1)
        except:
            self.fail("Can't init {} encoder".format(self._name))

    def test_pass_data(self):
        for in_channels in [1, 3]:
            for batch_size in [1, 3]:
                for img_size in [self._min_img_size + 64 * i for i in range(3)]:
                    for classes_num in range(1, 3):
                        with self.subTest(in_channels=in_channels, batch_size=batch_size, img_size=img_size, classes_num=classes_num):
                            try:
                                encoder = self._init_unet(in_channels=in_channels, classes_num=classes_num)
                                res = encoder(torch.rand((batch_size, in_channels, img_size, img_size)))
                                self.assertEqual(res.size(), self._get_out_size_by_input_size(batch_size, classes_num, img_size, img_size))
                                if img_size // 2 >= self._min_img_size:
                                    res = encoder(torch.rand((batch_size, in_channels, img_size // 2, img_size)))
                                    self.assertEqual(res.size(), self._get_out_size_by_input_size(batch_size, classes_num, img_size // 2, img_size))
                                    res = encoder(torch.rand((batch_size, in_channels, img_size, img_size // 2)))
                                    self.assertEqual(res.size(), self._get_out_size_by_input_size(batch_size, classes_num, img_size, img_size // 2))
                            except:
                                self.fail("Encoder doesn't pass correct data")

    def test_jit(self):
        encoder = self._init_unet(in_channels=2, classes_num=2).eval()

        try:
            torch.jit.trace(encoder, torch.rand(1, 2, self._min_img_size, self._min_img_size))
        except:
            self.fail("Fail to trace model")

    def test_onnx(self):
        encoder = self._init_unet(in_channels=2, classes_num=3).eval()

        try:
            torch.onnx.export(encoder, torch.rand(1, 2, self._min_img_size, self._min_img_size), 'test_onnx.onnx', verbose=False)
            self.assertTrue(os.path.exists('test_onnx.onnx') and os.path.isfile('test_onnx.onnx'))
        except:
            self.fail("Fail to trace model")
        os.remove('test_onnx.onnx')


class ResNet18DecoderTest(BasicUnetTest):
    _min_img_size = 64
    _name = "ResNet18Decoder"
    _layers_num = 5

    def _init_unet(self, in_channels: int, classes_num: int):
        return UNetDecoder(ResNet18(in_channels), classes_num=classes_num)

    @staticmethod
    def _get_out_size_by_input_size(batch_size, classes_num, img_size_x, img_size_y):
        return torch.Size([batch_size, classes_num, img_size_x, img_size_y])


class ResNet34DecoderTest(BasicUnetTest):
    _min_img_size = 64
    _name = "ResNet18Decoder"
    _layers_num = 5

    def _init_unet(self, in_channels: int, classes_num: int):
        return UNetDecoder(ResNet34(in_channels), classes_num=classes_num)

    @staticmethod
    def _get_out_size_by_input_size(batch_size, classes_num, img_size_x, img_size_y):
        return torch.Size([batch_size, classes_num, img_size_x, img_size_y])


class ResNet50DecoderTest(BasicUnetTest):
    _min_img_size = 64
    _name = "ResNet18Decoder"
    _layers_num = 5

    def _init_unet(self, in_channels: int, classes_num: int):
        return UNetDecoder(ResNet50(in_channels), classes_num=classes_num)

    @staticmethod
    def _get_out_size_by_input_size(batch_size, classes_num, img_size_x, img_size_y):
        return torch.Size([batch_size, classes_num, img_size_x, img_size_y])


class ResNet101DecoderTest(BasicUnetTest):
    _min_img_size = 64
    _name = "ResNet101Decoder"
    _layers_num = 5

    def _init_unet(self, in_channels: int, classes_num: int):
        return UNetDecoder(ResNet101(in_channels), classes_num=classes_num)

    @staticmethod
    def _get_out_size_by_input_size(batch_size, classes_num, img_size_x, img_size_y):
        return torch.Size([batch_size, classes_num, img_size_x, img_size_y])


class ResNet152DecoderTest(BasicUnetTest):
    _min_img_size = 64
    _name = "ResNet152Decoder"
    _layers_num = 5

    def _init_unet(self, in_channels: int, classes_num: int):
        return UNetDecoder(ResNet152(in_channels), classes_num=classes_num)

    @staticmethod
    def _get_out_size_by_input_size(batch_size, classes_num, img_size_x, img_size_y):
        return torch.Size([batch_size, classes_num, img_size_x, img_size_y])


class MobileNetV2DecoderTest(BasicUnetTest):
    _min_img_size = 64
    _name = "MobileNetV2Encoder"
    _layers_num = 7

    def _init_unet(self, in_channels: int, classes_num: int):
        return UNetDecoder(MobileNetV2Encoder(in_channels), classes_num=classes_num)

    @staticmethod
    def _get_out_size_by_input_size(batch_size, classes_num, img_size_x, img_size_y):
        return torch.Size([batch_size, classes_num, img_size_x, img_size_y])
