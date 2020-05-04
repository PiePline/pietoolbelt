from pietoolbelt.models.encoders.common import BasicEncoder
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module


class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        return self.layer(x)


class UNetDecoder(Module):
    def __init__(self, encoder: BasicEncoder, classes_num: int):
        super().__init__()
        self._encoder = encoder
        self._encoder.collect_layers_outputs(True)

        params = encoder.get_layers_params()

        decoder_stages = []
        for idx, param in enumerate(params[1:]):
            decoder_stages.append(UNetDecoderBlock(param['filter_size'], params[max(idx, 0)]['filter_size'], param['kernel_size'], param['stride'], param['stride']))
        self.decoder_stages = nn.ModuleList(decoder_stages)

        self.bottlenecks = nn.ModuleList([ConvBottleneck(p['filter_size'] * 2, p['filter_size']) for p in reversed(params[:-1])])

        self.last_upsample = UNetDecoderBlock(params[0]['filter_size'], params[0]['filter_size'], params[0]['kernel_size'], params[0]['stride'], params[0]['padding'])
        self.final = nn.Conv2d(params[0]['filter_size'], classes_num, 3, padding=1)

    def forward(self, data):
        x = self._encoder(data)
        encoder_outputs = self._encoder.get_layers_outputs() + [x]

        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, encoder_outputs[rev_idx - 1])

        x = self.last_upsample(x)
        f = self.final(x)
        return f
