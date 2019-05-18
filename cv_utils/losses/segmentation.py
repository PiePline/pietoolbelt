import torch
from torch import Tensor
from torch.nn import Module

from cv_utils.losses.common import ComposedLoss
from cv_utils.metrics.torch.segmentation import dice, _multiclass_metric
from cv_utils.models.utils import Activation


class DiceLoss(Module):
    """
    Dice loss function

    Args:
        eps (float): smooth value. When eps == 1 it's named Smooth Dice loss
        activation (srt): the activation function, that applied to predicted values. See :class:`Activation` for possible values
    """
    def __init__(self, eps: float = 1, activation: str = None):
        super().__init__()
        self._activation = Activation(activation)
        self._eps = eps

    def forward(self, output: Tensor, target: Tensor):
        return 1 - dice(self._activation(output), target, eps=self._eps)


class BCEDiceLoss(ComposedLoss):
    """
    Dice loss function

    Args:
        bce_w (float): weight of bce loss
        dice_w (float): weight of dice loss
        eps (float): smooth value. When eps == 1 it's named Smooth Dice loss
        activation (srt): the activation function, that applied to predicted values. See :class:`Activation` for possible values
    """
    def __init__(self, bce_w: float, dice_w: float, eps: float = 1, activation: str = None):
        bce_loss = torch.nn.BCEWithLogitsLoss()
        dice_loss = DiceLoss(eps=eps, activation=activation)

        super().__init__([bce_loss, dice_loss], [bce_w, dice_w])


class MulticlassSegmentationLoss(Module):
    """
    Wrapper loss function to work with multiclass segmentation.
    This just split masks by classes and calculate :arg:`base_loss` for every class. After that all loss values summarized

    Args:
         base_loss (Module): basic loss object
    """
    def __init__(self, base_loss: Module):
        super().__init__()
        self._base_loss = base_loss

    def forward(self, output, target):
        return _multiclass_metric(self._base_loss, output, target)
