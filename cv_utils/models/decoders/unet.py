from cv_utils.models.encoders.common import BasicEncoder
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
        print(dec.size(), enc.size())
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

        filters = encoder.get_layers_params()

        decoder_stages = []
        for idx in range(1, len(filters)):
            decoder_stages.append(UNetDecoderBlock(filters[idx]['filter_size'], filters[max(idx - 1, 0)]['filter_size'], filters[idx]['kernel_size'], filters[idx]['stride'], filters[idx]['stride']))
        self.decoder_stages = nn.ModuleList(decoder_stages)

        self.bottlenecks = nn.ModuleList([ConvBottleneck(f['filter_size'] * 2, f['filter_size']) for f in reversed(filters[:-1])])

        self.last_upsample = UNetDecoderBlock(filters[0]['filter_size'], filters[0]['filter_size'], filters[0]['kernel_size'], filters[0]['stride'], filters[0]['padding'])
        self.final = nn.Conv2d(filters[0]['filter_size'], classes_num, 3, padding=1)

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
