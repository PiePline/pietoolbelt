import os
import unittest

import torch
from torch.nn import Module

from pietoolbelt.models.decoders.ssd import SSDDecoder
from pietoolbelt.models.encoders.inception import InceptionV3Encoder
from pietoolbelt.models.encoders.mobile_net import MobileNetV2Encoder
from pietoolbelt.models.encoders.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

__all__ = ['ResNet18DecoderTest']


class BasicUnetTest(unittest.TestCase):
    def _init_ssd(self, in_channels: int, classes_num: int) -> Module:
        raise NotImplemented()

    _min_img_size = None
    _name = None
    _layers_num = None
    _get_out_size_by_input_size = None

    def test_init(self):
        try:
            encoder = self._init_ssd(3, 1)
        except:
            self.fail("Can't init {} encoder".format(self._name))

    def test_pass_data(self):
        for in_channels in [1, 3]:
            for batch_size in [1, 3]:
                for img_size in [self._min_img_size + 64 * i for i in range(3)]:
                    for classes_num in range(1, 3):
                        with self.subTest(in_channels=in_channels, batch_size=batch_size, img_size=img_size, classes_num=classes_num):
                            try:
                                encoder = self._init_ssd(in_channels=in_channels, classes_num=classes_num)
                                res = encoder(torch.rand((batch_size, in_channels, img_size, img_size)))
                                self.assertEqual(res.size(),
                                                 self._get_out_size_by_input_size(batch_size, img_size, img_size))
                                if img_size // 2 >= self._min_img_size:
                                    res = encoder(torch.rand((batch_size, in_channels, img_size // 2, img_size)))
                                    self.assertEqual(res.size(),
                                                     self._get_out_size_by_input_size(batch_size, img_size // 2, img_size))
                                    res = encoder(torch.rand((batch_size, in_channels, img_size, img_size // 2)))
                                    self.assertEqual(res.size(), self._get_out_size_by_input_size(batch_size, img_size, img_size // 2))
                            except Exception as err:
                                self.fail("Encoder doesn't pass correct data. Error msg: [{}]".format(err))

    def test_jit(self):
        encoder = self._init_ssd(in_channels=2, classes_num=1).eval()

        try:
            torch.jit.trace(encoder, torch.rand(1, 2, self._min_img_size, self._min_img_size))
        except:
            self.fail("Fail to trace model")

    def test_onnx(self):
        encoder = self._init_ssd(in_channels=2).eval()

        try:
            torch.onnx.export(encoder, torch.rand(1, 2, self._min_img_size, self._min_img_size), 'test_onnx.onnx', verbose=False)
            self.assertTrue(os.path.exists('test_onnx.onnx') and os.path.isfile('test_onnx.onnx'))
        except:
            self.fail("Fail to trace model")
        os.remove('test_onnx.onnx')


class ResNet18DecoderTest(BasicUnetTest):
    _min_img_size = 512
    _name = "ResNet18Decoder"
    _layers_num = 5

    def _init_ssd(self, in_channels: int, classes_num: int):
        return SSDDecoder(ResNet18(in_channels), classes_num=classes_num)

    @staticmethod
    def _get_out_size_by_input_size(batch_size, classes_num, img_size_x, img_size_y):
        return torch.Size([batch_size, classes_num, img_size_x, img_size_y])
