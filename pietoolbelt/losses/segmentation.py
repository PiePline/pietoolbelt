import torch
from torch import Tensor
from torch.nn import Module

from pietoolbelt.losses.common import ComposedLoss, Reduction
from pietoolbelt.metrics.torch.segmentation import dice, _multiclass_metric, _split_masks_by_classes
from pietoolbelt.models.utils import Activation

__all__ = ['DiceLoss', 'BCEDiceLoss', 'LovaszSoftmax']


class DiceLoss(Module):
    """
    Dice loss function

    Args:
        eps (float): smooth value. When eps == 1 it's named Smooth Dice loss
        activation (srt): the activation function, that applied to predicted values. See :class:`Activation` for possible values
    """

    def __init__(self, eps: float = 1, activation: str = None, reduction: Reduction = Reduction('mean')):
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

    def __init__(self, bce_w: float, dice_w: float, eps: float = 1, activation: str = None, reduction: Reduction = Reduction('mean'),
                 class_weights: [] = None):
        if class_weights is None:
            bce_loss = torch.nn.BCELoss()
        else:
            bce_loss = torch.nn.BCELoss(torch.Tensor(class_weights))
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


class LovaszSoftmax(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction

    @staticmethod
    def prob_flatten(input, target):
        assert input.dim() in [4, 5]
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            if num_classes == 1:
                input_c = inputs[:, 0]
            else:
                input_c = inputs[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(self.lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs, targets):
        inputs, targets = self.prob_flatten(inputs, targets)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses

    @staticmethod
    def lovasz_grad(gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard
