import torch
from piepline import AbstractMetric, MetricsProcessor, MetricsGroup
from torch import Tensor, nn
import numpy as np

from pietoolbelt.models.utils import Activation

__all__ = ['dice', 'jaccard', 'multiclass_dice', 'multiclass_jaccard',
           'DiceMetric', 'JaccardMetric', 'SegmentationMetricsProcessor',
           'MulticlassDiceMetric', 'MulticlassJaccardMetric', 'MulticlassSegmentationMetricsProcessor']


def _split_masks_by_classes(pred: Tensor, target: Tensor) -> []:
    """
    Split masks by classes

    Args:
        pred (Tensor): predicted masks of shape [B, C, H, W]
        target (Tensor): target masks of shape [B, C, H, W]

    Returns:
        List: list of masks pairs [pred, target], splitted by channels. List shape: [C, 2, B, H, W]
    """
    preds = torch.split(pred, 1, dim=1)
    targets = torch.split(target, 1, dim=1)

    return list(zip(preds, targets))


def dice(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Calculate Dice coefficient

    Args:
        pred (Tensor): predicted masks of shape [B, 1, H, W]
        target (Tensor): target masks of shape [B, 1, H, W]
        eps (float): smooth value

    Returns:
        Tensor: Tensor with values of Dice coefficient. Tensor size: [B]
    """
    pred_inner = pred.view((pred.size(0), pred.size(2) * pred.size(3)))
    target_inner = target.view((target.size(0), target.size(2) * target.size(3)))

    intersection = (pred_inner * target_inner).sum(1)
    return (2. * intersection + eps) / ((pred_inner * pred_inner).sum(1) + (target_inner * target_inner).sum(1) + eps)


def jaccard(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Calculate Jaccard coefficient

    Args:
        pred (Tensor): predicted masks of shape [B, 1, H, W]
        target (Tensor): target masks of shape [B, 1, H, W]
        eps (float): smooth value

    Returns:
        Tensor: Tensor with values of Jaccard coefficient. Tensor size: [B]
    """
    preds_inner = pred.view((pred.size(0), pred.size(2) * pred.size(3)))
    trues_inner = target.view((target.size(0), target.size(2) * target.size(3)))

    intersection = (preds_inner * trues_inner).sum(1)
    return (intersection + eps) / ((preds_inner + trues_inner).sum(1) - intersection + eps)


def _multiclass_metric(func: callable, pred, target) -> Tensor:
    res = torch.zeros((pred.shape[1], pred.shape[0]), dtype=pred.dtype)
    for i, [p, t] in enumerate(_split_masks_by_classes(pred, target)):
        res[i] = func(torch.squeeze(p, dim=1), torch.squeeze(t, dim=1))
    return res


def multiclass_dice(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Calculate Dice coefficient for multiclass case

    Args:
        pred (Tensor): predicted masks of shape [B, C, H, W]
        target (Tensor): target masks of shape [B, C, H, W]
        eps (float): smooth value

    Returns:
        Tensor: Tensor with values of Dice coefficient. Tensor size: [C, B]
    """
    return _multiclass_metric(lambda out, tar: dice(out, tar, eps), pred, target)


def multiclass_jaccard(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Calculate Jaccard coefficient for multiclass case

    Args:
        pred (Tensor): predicted masks of shape [B, C, H, W]
        target (Tensor): target masks of shape [B, C, H, W]
        eps (float): smooth value

    Returns:
        Tensor: Tensor with values of Jaccard coefficient. Tensor size: [C, B]
    """
    return _multiclass_metric(lambda out, tar: jaccard(out, tar, eps), pred, target)


class _SegmentationMetric(AbstractMetric):
    def __init__(self, name: str, func: callable, activation: str = None, eps: float = 1e-7, threshold: float = None):
        super().__init__(name)
        self._func = func
        self._activation = Activation(activation)
        self._eps = eps

        if threshold is None:
            self.tensor_preproc = lambda x: x
        else:
            self.tensor_preproc = lambda x: self._thresh(x, threshold)

    def calc(self, output: Tensor, target: Tensor) -> np.ndarray:
        return np.squeeze(self._func(self._activation(output), target, self._eps).cpu().numpy())

    @staticmethod
    def _thresh(output: Tensor, thresh) -> Tensor:
        output[output < thresh] = 0
        output[output > 0] = 1
        return output

    @staticmethod
    def min_val() -> float:
        return 0

    @staticmethod
    def max_val() -> float:
        return 1


class MulticlassSegmentationMetric(_SegmentationMetric):
    def __init__(self, name: str, func: callable, activation: str = None, reduction: str = None):
        super().__init__(name, func, activation)

        if reduction is None:
            self._reduction = lambda x: x
        elif reduction == 'sum':
            self._reduction = lambda x: x.sum(0)
        elif reduction == 'mean':
            self._reduction = lambda x: x.mean(0)
        else:
            raise Exception("Unexpected reduction '{}'. Possible values: [sum, mean]".format(reduction))

    def calc(self, output: Tensor, target: Tensor) -> np.ndarray:
        res = np.squeeze(self._func(self._activation(output), target).data.cpu().numpy())
        return self._reduction(res)


class DiceMetric(_SegmentationMetric):
    def __init__(self, activation: str = None, eps: float = 1e-7, thresh: float = None):
        super().__init__('dice' if thresh is None else 'dice_{:1.1f}'.format(thresh), dice, activation, eps, threshold=thresh)


class JaccardMetric(_SegmentationMetric):
    def __init__(self, activation: str = None, eps: float = 1e-7, thresh: float = None):
        super().__init__('jaccard' if thresh is None else 'jaccard_{:1.1f}'.format(thresh), jaccard, activation, eps, threshold=thresh)


class SegmentationMetricsProcessor(MetricsProcessor):
    def __init__(self, stage_name: str, activation: str = None, thresholds: [float] = None):
        super().__init__()
        group = MetricsGroup(stage_name)

        if thresholds is not None:
            for thresh in thresholds:
                if thresh is not None:
                    group.add(JaccardMetric(activation, thresh=thresh)).add(DiceMetric(activation, thresh=thresh))
        self.add_metrics_group(group.add(JaccardMetric(activation)).add(DiceMetric(activation)))


class MulticlassDiceMetric(MulticlassSegmentationMetric):
    def __init__(self, activation: str = None, reduction: str = 'sum'):
        super().__init__('dice', func=multiclass_dice, activation=activation, reduction=reduction)


class MulticlassJaccardMetric(MulticlassSegmentationMetric):
    def __init__(self, activation: str = None, reduction: str = 'sum'):
        super().__init__('jaccard', func=multiclass_jaccard, activation=activation, reduction=reduction)


class MulticlassSegmentationMetricsProcessor(MetricsProcessor):
    def __init__(self, stage_name: str, activation: str = None, reduction: str = 'sum'):
        super().__init__()
        self.add_metrics_group(MetricsGroup(stage_name).add(MulticlassJaccardMetric(activation=activation, reduction=reduction))
                               .add(MulticlassDiceMetric(activation=activation, reduction=reduction)))
