from abc import ABCMeta, abstractmethod
from random import randint
from typing import Tuple

import numpy as np
from albumentations import HorizontalFlip, VerticalFlip, Rotate, CLAHE, GaussNoise


class AbstractTTA(metaclass=ABCMeta):
    @abstractmethod
    def preprocess(self, data: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, target: np.ndarray):
        pass


class HFlipTTA(AbstractTTA):
    flip = HorizontalFlip(p=1)

    def preprocess(self, data: np.ndarray):
        return self.flip(image=data)['image']

    def postprocess(self, target: np.ndarray):
        return self.flip(image=target)['image']


class VFlipTTA(AbstractTTA):
    flip = VerticalFlip(p=1)

    def preprocess(self, data: np.ndarray):
        return self.flip(image=data)['image']

    def postprocess(self, target: np.ndarray):
        return self.flip(image=target)['image']


class RotateTTA(AbstractTTA):
    rotate = Rotate(p=1)

    def __init__(self, angle_range: Tuple[float, float] = (-180, 180)):
        self._angle_range = angle_range
        self._last_angle = None

    def preprocess(self, data: np.ndarray):
        self._last_angle = randint(self._angle_range[0], self._angle_range[1])
        return self.rotate.apply(img=data, angle=self._last_angle)

    def postprocess(self, target: np.ndarray):
        return self.rotate.apply(img=target, angle=-self._last_angle)


class CLAHETTA(AbstractTTA):
    clahe = CLAHE(p=1)

    def preprocess(self, data: np.ndarray):
        return self.clahe(image=data)['image']

    def postprocess(self, target: np.ndarray):
        return target


class GaussNoiseTTA(AbstractTTA):
    noise = GaussNoise(p=1)

    def preprocess(self, data: np.ndarray):
        return self.noise(image=data)['image']

    def postprocess(self, target: np.ndarray):
        return target
