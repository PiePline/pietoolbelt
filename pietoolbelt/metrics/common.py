import torch
import numpy as np
from piepline import AbstractMetric


def dice2jaccard(dice_val):
    """
    Calc Jaccard coefficient from Dice

    :param dice_val: value of dice coefficient
    :type: float or np.ndarray
    :return: float or np.ndarray corresponds of input type
    """
    return dice_val / (2 - dice_val)


def jaccard2dice(jaccard_val):
    """
    Calc Dice coefficient from Jaccard

    Args:
        jaccard_val (float or np.ndarray): value of Jaccard coefficient

    Returns:
        Return np.ndarray or torch.Tensor in depends of input argument type
    """
    return 2 * jaccard_val / (1 + jaccard_val)


class ScalarMetric(AbstractMetric):
    def __init__(self, name: str, param: torch.nn.Parameter):
        super().__init__(name)
        self.param = param

    def calc(self, output: torch.Tensor, target: torch.Tensor) -> np.ndarray or float:
        return self.param.data
