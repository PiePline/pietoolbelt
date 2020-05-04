import numpy as np


def rmse(predict: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.sqrt(np.mean((predict - target) ** 2, axis=0))))


def amad(predict: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.mean(np.abs(predict - target), axis=0)))


def relative(predict: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.mean(np.abs(predict - target) / (target + 1e-6), axis=1)))
