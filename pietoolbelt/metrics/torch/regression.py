import torch
import numpy as np
from piepline import AbstractMetric
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor

from pietoolbelt.metrics.cpu.regression import rmse as rmse_cpu
from pietoolbelt.metrics.cpu.regression import amad as amad_cpu
from pietoolbelt.metrics.cpu.regression import relative as relative_cpu

__all__ = ['rmse', 'amad', 'relative', 'AMADMetric', 'RelativeMetric', 'RMSEMetric']


def rmse(predict: Tensor, target: Tensor) -> float:
    return float(torch.mean(torch.sqrt(torch.mean((predict - target) ** 2, axis=0))).cpu())


def amad(predict: Tensor, target: Tensor) -> float:
    return float(torch.mean(torch.mean(torch.abs(predict - target), axis=0)).cpu())


def relative(predict: Tensor, target: Tensor) -> float:
    return float(torch.mean(torch.mean(torch.abs(predict - target) / (target + 1e-6), axis=0)).cpu())


class _AbstractRegressionMetric(AbstractMetric):
    def __init__(self, name: str, calc_cpu: callable, calc_torch: callable, min_max_scaler: MinMaxScaler = None):
        super().__init__(name)
        self.scaler = min_max_scaler

        self._calc_cpu, self._calc_torch = calc_cpu, calc_torch
        if self.scaler is None:
            self._calc = self._calc_without_scaler
        else:
            self._calc = self._calc_with_scaler

    def _calc_with_scaler(self, output: Tensor, target: Tensor) -> np.ndarray or float:
        output_inner = self.scaler.inverse_transform(output.detach().cpu().numpy())
        target_inner = self.scaler.inverse_transform(target.detach().cpu().numpy())
        return self._calc_cpu(output_inner, target_inner)

    def _calc_without_scaler(self, output: Tensor, target: Tensor) -> np.ndarray or float:
        return self._calc_torch(output, target)

    def calc(self, output: Tensor, target: Tensor) -> np.ndarray or float:
        return self._calc(output, target)


class AMADMetric(_AbstractRegressionMetric):
    def __init__(self, min_max_scaler: MinMaxScaler = None):
        super().__init__("AMAD", calc_cpu=amad_cpu, calc_torch=amad, min_max_scaler=min_max_scaler)


class RMSEMetric(_AbstractRegressionMetric):
    def __init__(self, min_max_scaler: MinMaxScaler = None):
        super().__init__("RMSE", calc_cpu=rmse_cpu, calc_torch=rmse, min_max_scaler=min_max_scaler)


class RelativeMetric(_AbstractRegressionMetric):
    def __init__(self, min_max_scaler: MinMaxScaler = None):
        super().__init__("Relative", calc_cpu=relative_cpu, calc_torch=relative, min_max_scaler=min_max_scaler)
