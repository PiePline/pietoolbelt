import torch
from torch import Tensor
from torch.nn import Module

from cv_utils.losses.common import ComposedLoss, Reduction
from cv_utils.metrics.torch.segmentation import dice, _multiclass_metric, _split_masks_by_classes
from cv_utils.models.utils import Activation


class DiceLoss(Module):
    """
    Dice loss function

    Args:
        eps (float): smooth value. When eps == 1 it's named Smooth Dice loss
        activation (srt): the activation function, that applied to predicted values. See :class:`Activation` for possible values
    """
    def __init__(self, eps: float = 1, activation: str = None, reduction: Reduction = Reduction('sum')):
        super().__init__()
        self._activation = Activation(activation)
        self._reduction = reduction
        self._eps = eps

    def forward(self, output: Tensor, target: Tensor):
        return self._reduction(1 - dice(self._activation(output), target, eps=self._eps))


class BCEDiceLoss(ComposedLoss):
    """
    Dice loss function

    Args:
        bce_w (float): weight of bce loss
        dice_w (float): weight of dice loss
        eps (float): smooth value. When eps == 1 it's named Smooth Dice loss
        activation (srt): the activation function, that applied to predicted values. See :class:`Activation` for possible values
    """
    def __init__(self, bce_w: float, dice_w: float, eps: float = 1, activation: str = None, reduction: Reduction = None):
        bce_loss = torch.nn.BCEWithLogitsLoss()
        dice_loss = DiceLoss(eps=eps, activation=activation, reduction=reduction)

        super().__init__([bce_loss, dice_loss], [bce_w, dice_w])


class MulticlassSegmentationLoss(Module):
    """
    Wrapper loss function to work with multiclass segmentation.
    This just split masks by classes and calculate :arg:`base_loss` for every class. After that all loss values summarized

    Args:
         base_loss (Module): basic loss object
    """
    def __init__(self, base_loss: Module, reduction: Reduction = Reduction('sum')):
        super().__init__()
        self._base_loss = base_loss
        self._reduction = reduction

    def forward(self, output: Tensor, target: Tensor):
        res = torch.zeros((output.size(1)), dtype=output.dtype)
        for i, [p, t] in enumerate(_split_masks_by_classes(output, target)):
            res[i] = self._base_loss(torch.squeeze(p, dim=1), torch.squeeze(t, dim=1))
        return self._reduction(res)
