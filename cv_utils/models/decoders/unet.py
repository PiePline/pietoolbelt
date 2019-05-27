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
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
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

        encoder_res = self._encoder(torch.rand(1, list(self._encoder.parameters())[0].size(1), 128, 128))
        encoder_res = [encoder_res] + self._encoder.get_layers_outputs()

        filters = [r.size(1) for r in encoder_res]
        filters = filters[1:] + filters[:1]

        self.bottlenecks = nn.ModuleList([ConvBottleneck(f * 2, f) for f in reversed(filters[:-1])])
        self.decoder_stages = nn.ModuleList([UnetDecoderBlock(filters[idx], filters[max(idx - 1, 0)]) for idx in range(1, len(filters))])

        self.last_upsample = UnetDecoderBlock(filters[0], filters[0])
        self.final = nn.Conv2d(filters[0], classes_num, 3, padding=1)

    def forward(self, data):
        x = self._encoder(data)
        encoder_outputs = [x] + self._encoder.get_layers_outputs()

        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, encoder_outputs[rev_idx])

        x = self.last_upsample(x)
        f = self.final(x)
        return f
