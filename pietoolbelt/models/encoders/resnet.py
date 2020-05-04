import torch.nn as nn
from pietoolbelt.models.encoders.common import BasicEncoder

__all__ = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(BasicEncoder):
    """
    ResNetN encoder.

    This model get tensor of size `[B, C, 32 * i, 32 * j]`, where i, j = 2, 3, 4, ...
    The produce tensor size is:
    * `[B, 512, i, j]` for `BasicBlock`
    * `[B, 2048, i, j]` for `Bottleneck`

    Args:
        block (Module): basic block, that will be used for construct layers
        layers (list): list of number of blocks for each layer
        in_channels (int): number of channels for input image
        zero_init_residual (bool): is need to init the last BN in each residual branch by zeros
    """

    def __init__(self, block, layers, in_channels: int, zero_init_residual=False):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self._init_layers_params(block)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self._process_layer_output(x)

        x = self.maxpool(x)
        x = self.layer1(x)
        x = self._process_layer_output(x)

        x = self.layer2(x)
        x = self._process_layer_output(x)

        x = self.layer3(x)
        x = self._process_layer_output(x)

        x = self.layer4(x)

        return x

    def get_layers_params(self) -> []:
        return self._layers_params

    def _init_layers_params(self, basic_block):
        if basic_block is BasicBlock:
            self._layers_params = [{'filter_size': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                                   {'filter_size': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                                   {'filter_size': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                                   {'filter_size': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                                   {'filter_size': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1}]
        elif basic_block is Bottleneck:
            self._layers_params = [{'filter_size': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                                   {'filter_size': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                                   {'filter_size': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                                   {'filter_size': 1024, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                                   {'filter_size': 2048, 'kernel_size': 3, 'stride': 1, 'padding': 1}]
        else:
            raise Exception("Undefined basic block")


class ResNetBasicBlock(ResNet):
    def __init__(self, layers, in_channels: int, zero_init_residual=False):
        super().__init__(BasicBlock, layers, in_channels=in_channels, zero_init_residual=zero_init_residual)

    def _init_layers_params(self, basic_block):
        self._layers_params = [{'filter_size': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                               {'filter_size': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                               {'filter_size': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                               {'filter_size': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                               {'filter_size': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1}]


class ResNetBottleneck(ResNet):
    def __init__(self, layers, in_channels: int, zero_init_residual=False):
        super().__init__(Bottleneck, layers, in_channels=in_channels, zero_init_residual=zero_init_residual)

    def _init_layers_params(self, basic_block):
        self._layers_params = [{'filter_size': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                               {'filter_size': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                               {'filter_size': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                               {'filter_size': 1024, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                               {'filter_size': 2048, 'kernel_size': 3, 'stride': 1, 'padding': 1}]


class ResNet18(ResNetBasicBlock):
    def __init__(self, in_channels: int, zero_init_residual=False):
        super().__init__([2, 2, 2, 2], in_channels, zero_init_residual)


class ResNet34(ResNetBasicBlock):
    def __init__(self, in_channels: int, zero_init_residual=False):
        super().__init__([3, 4, 6, 3], in_channels, zero_init_residual)


class ResNet50(ResNetBottleneck):
    def __init__(self, in_channels: int, zero_init_residual=False):
        super().__init__([3, 4, 6, 3], in_channels, zero_init_residual)


class ResNet101(ResNetBottleneck):
    def __init__(self, in_channels: int, zero_init_residual=False):
        super().__init__([3, 4, 23, 3], in_channels, zero_init_residual)


class ResNet152(ResNetBottleneck):
    def __init__(self, in_channels: int, zero_init_residual=False):
        super().__init__([3, 8, 36, 3], in_channels, zero_init_residual)
