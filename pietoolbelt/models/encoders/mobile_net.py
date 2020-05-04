from pietoolbelt.models.encoders.common import BasicEncoder

import torch.nn as nn
import math

__all__ = ['MobileNetV2Encoder']


def conv_bn(in_channels: int, out_channels: int, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2Encoder(BasicEncoder):
    """
    MobileNet-V2 encoder

    This model get tensor of size `[B, C, 32 * i, 32 * j]`, where i, j = 2, 3, 4, ... and produce tensor of size `[B, 1280 * width_mult, i, j]`

    Args:
        in_channels (int): number of channels for input image
        width_mult (float): output tensor width coefficient. Min val: 0.0625
    """
    def __init__(self, in_channels: int = 3, width_mult=1.):
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # expand_ratio, output_channels, blocks_num, stride
            [1, 16, 1, 1],  # 1
            [6, 24, 2, 2],  # 2
            [6, 32, 3, 2],  # 3
            [6, 64, 4, 2],  # 4
            [6, 96, 3, 1],  # 5
            [6, 160, 3, 2],  # 6
            [6, 320, 1, 1],  # 7
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel

        self.conv1 = conv_bn(in_channels, input_channel, 2)

        # building inverted residual blocks
        for layer_id, [t, c, n, s] in enumerate(interverted_residual_setting):
            output_channel = int(c * width_mult)
            for i in range(n):
                cur_block = block(input_channel, output_channel, s if i == 0 else 1, expand_ratio=t)
                setattr(self, 'layer{}_{}'.format(layer_id + 1, i + 1), cur_block)
                input_channel = output_channel

        # building last several layers
        self.last_layer = conv_1x1_bn(input_channel, last_channel)

        self._initialize_weights()

    def _forward(self, x):
        x = self.conv1(x)

        x = self._process_layer_output(self.layer1_1(x))

        x = self.layer2_1(x)
        x = self._process_layer_output(self.layer2_2(x))

        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self._process_layer_output(self.layer3_3(x))

        x = self.layer4_1(x)
        x = self.layer4_2(x)
        x = self.layer4_3(x)
        x = self._process_layer_output(self.layer4_4(x))

        x = self.layer5_1(x)
        x = self.layer5_2(x)
        x = self._process_layer_output(self.layer5_3(x))

        x = self.layer6_1(x)
        x = self.layer6_2(x)
        x = self._process_layer_output(self.layer6_3(x))

        x = self._process_layer_output(self.layer7_1(x))

        return self.last_layer(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @staticmethod
    def get_layers_params() -> []:
        return [{'filter_size': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'filter_size': 24, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'filter_size': 32, 'kernel_size': 3, 'stride': 1, 'padding': 2},
                {'filter_size': 64, 'kernel_size': 3, 'stride': 1, 'padding': 2},
                {'filter_size': 96, 'kernel_size': 5, 'stride': 2, 'padding': 1},
                {'filter_size': 160, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'filter_size': 320, 'kernel_size': 5, 'stride': 2, 'padding': 0},
                {'filter_size': 1280, 'kernel_size': 5, 'stride': 2, 'padding': 0}]
