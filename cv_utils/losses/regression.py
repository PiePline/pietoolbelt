import torch
from torch import nn


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._mse = nn.MSELoss()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        return torch.sqrt(self._mse(output, target))
