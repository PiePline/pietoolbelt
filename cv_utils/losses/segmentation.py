import torch
from torch.nn import Module

from cv_utils.metrics.numpy.segmentation import split_masks_by_classes, dice
from cv_utils.models.utils import Activation


class DiceLoss(Module):
    def __init__(self, eps: float = 1e-7, activation: str = None):
        super().__init__()
        self._activation = Activation(activation)
        self._eps = eps

    def forward(self, output, target):
        return 1 - dice(self._activation(output), target, eps=self._eps)


class ComposedLoss(Module):
    def __init__(self, losses: [], coeffs: [] = None):
        super().__init__()
        self._losses = losses
        if coeffs is None:
            self._coeffs = [1 / len(losses) for _ in losses]
        else:
            if len(coeffs) != len(losses):
                raise Exception("Number of coefficients ({}) doesn't equal to number of losses ({})".format(len(coeffs), len(losses)))
            self._coeffs = coeffs

        self._apply_weight = lambda l, c: c * l

    @staticmethod
    def _multiply_coeff_with_exp(loss_val, weight):
        return torch.exp(-weight) * loss_val + weight

    def enable_learn_coeffs(self, strategy: str = 'exp'):
        for i, c in enumerate(self._coeffs):
            self._coeffs[i] = torch.nn.Parameter(torch.Tensor(np.array([c], dtype=np.float32)), requires_grad=True)

        if strategy == 'exp':
            self._apply_weight = self._multiply_coeff_with_exp
        else:
            raise Exception("Strategy '{}' doesn't have implementation. ComposedLoss have only 'exp'".format(strategy))

    def get_coeffs(self) -> []:
        return self._coeffs

    def forward(self, *args, **kwargs):
        res = 0

        for l, c in zip(self._losses, self._coeffs):
            res += self._apply_weight(l(*args, **kwargs), c)

        return res


class MulticlassSegmentationLoss(Module):
    def __init__(self, base_loss):
        super().__init__()
        self._base_loss = base_loss

    def forward(self, output, target):
        overall_res = []
        for p, t in split_masks_by_classes(output, target):
            overall_res.append(self._base_loss(p, t))
        return sum(overall_res)


class BCEDiceLoss(Module):
    def __init__(self, bce_w: float, dice_w: float):
        super().__init__()

        bce = torch.nn.BCEWithLogitsLoss()
        dice = DiceLoss(eps=1e-7)

        self._loss = MulticlassSegmentationLoss(ComposedLoss([bce, dice], [bce_w, dice_w]))

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        return self._loss(output, target)
