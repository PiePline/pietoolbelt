import torch
from neural_pipeline import AbstractMetric, MetricsProcessor, MetricsGroup
from torch import Tensor
import numpy as np

from cv_utils.models.utils import Activation


def split_masks_by_classes(pred: Tensor, target: Tensor):
    preds = torch.split(pred, 1, dim=1)
    targets = torch.split(target, 1, dim=1)

    return list(zip(preds, targets))


def dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7):
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)

    intersection = (iflat * tflat).sum()

    res = (2. * intersection + eps) / (torch.sum(iflat * iflat) + torch.sum(tflat * tflat) + eps)

    return res


def jaccard(preds: torch.Tensor, trues: torch.Tensor, eps: float = 1e-7):
    preds_inner = preds.cpu().data.numpy().copy()
    trues_inner = trues.cpu().data.numpy().copy()

    preds_inner = np.reshape(preds_inner, (preds_inner.shape[0], preds_inner.size // preds_inner.shape[0]))
    trues_inner = np.reshape(trues_inner, (trues_inner.shape[0], trues_inner.size // trues_inner.shape[0]))
    intersection = (preds_inner * trues_inner).sum(1)
    scores = (intersection + eps) / ((preds_inner + trues_inner).sum(1) - intersection + eps)

    return scores


def multiclass_dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7):
    res, num = 0, 0
    for p, t in split_masks_by_classes(pred, target):
        res += dice(p, t, eps)
        num += 1
    return res / num


def multiclass_jaccard(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7):
    res, num = 0, 0
    for p, t in split_masks_by_classes(pred, target):
        res += jaccard(p, t, eps)
        num += 1
    return res / num


class DiceMetric(AbstractMetric):
    def __init__(self, activation: str = None):
        super().__init__('dice')
        self._activation = Activation(activation)

    def calc(self, output: torch.Tensor, target: torch.Tensor) -> np.ndarray or float:
        return multiclass_dice(self._activation(output), target)

    @staticmethod
    def min_val() -> float:
        return 0

    @staticmethod
    def max_val() -> float:
        return 1


class JaccardMetric(AbstractMetric):
    def __init__(self, activation: str = None):
        super().__init__('jaccard')
        self._activation = Activation(activation)

    def calc(self, output: torch.Tensor, target: torch.Tensor) -> np.ndarray or float:
        return multiclass_jaccard(self._activation(output), target)

    @staticmethod
    def min_val() -> float:
        return 0

    @staticmethod
    def max_val() -> float:
        return 1


class ScalarMetric(AbstractMetric):
    def __init__(self, name: str, param: torch.nn.Parameter):
        super().__init__(name)
        self.param = param

    def calc(self, output: torch.Tensor, target: torch.Tensor) -> np.ndarray or float:
        return self.param.data


class SegmentationMetricsProcessor(MetricsProcessor):
    def __init__(self, stage_name: str):
        super().__init__()
        self.add_metrics_group(MetricsGroup(stage_name).add(JaccardMetric(activation='sigmoid')).add(DiceMetric(activation='sigmoid')))
