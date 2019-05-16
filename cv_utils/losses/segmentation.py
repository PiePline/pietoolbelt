import torch
from torch import Tensor
from torch.nn import Module

from cv_utils.losses.common import ComposedLoss
from cv_utils.metrics.torch.segmentation import _split_masks_by_classes, dice
from cv_utils.models.utils import Activation


class DiceLoss(Module):
    def __init__(self, eps: float = 1, activation: str = None):
        super().__init__()
        self._activation = Activation(activation)
        self._eps = eps

    def forward(self, output: Tensor, target: Tensor):
        return 1 - dice(self._activation(output), target, eps=self._eps)


class BCEDiceLoss(Module):
    def __init__(self, bce_w: float, dice_w: float):
        super().__init__()

        bce = torch.nn.BCEWithLogitsLoss()
        dice = DiceLoss()

        self._loss = MulticlassSegmentationLoss(ComposedLoss([bce, dice], [bce_w, dice_w]))

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        return self._loss(output, target)


class MulticlassSegmentationLoss(Module):
    def __init__(self, base_loss):
        super().__init__()
        self._base_loss = base_loss

    def forward(self, output, target):
        overall_res = 0
        for p, t in _split_masks_by_classes(output, target):
            overall_res += self._base_loss(p, t)
        return overall_res
