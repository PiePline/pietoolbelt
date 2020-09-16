from abc import ABCMeta, abstractmethod

import cv2
import numpy as np

__all__ = ['AbstractVisualizer', 'ColormapVisualizer', 'MulticlassColormapVisualizer', 'DetectionVisualiser']


class AbstractVisualizer(metaclass=ABCMeta):
    @abstractmethod
    def process_img(self, image: np.ndarray, target: any) -> np.ndarray:
        """
        Visualize target on image

        Args:
            image: image array with shape (C, H, W)
            target: target object

        Returns:
            image
        """


class ColormapVisualizer(AbstractVisualizer):
    def __init__(self, proportions: [float, float], colormap=cv2.COLORMAP_JET):
        self._proportions = proportions
        self._colormap = colormap

    def process_img(self, image, mask) -> np.ndarray:
        heatmap_img = cv2.applyColorMap(mask, self._colormap)
        res = cv2.addWeighted(heatmap_img, self._proportions[1], image, self._proportions[0], 0)

        if len(image.shape) > 2:
            res[:, :, 0] = np.where(mask > 0, res[:, :, 0], image[:, :, 0])
            res[:, :, 1] = np.where(mask > 0, res[:, :, 1], image[:, :, 1])
            res[:, :, 2] = np.where(mask > 0, res[:, :, 2], image[:, :, 2])
        else:
            res[:, :, 0] = np.where(mask > 0, res[:, :, 0], image[:, :])
            res[:, :, 1] = np.where(mask > 0, res[:, :, 1], image[:, :])
            res[:, :, 2] = np.where(mask > 0, res[:, :, 2], image[:, :])

        return res


class ContourVisualizer(AbstractVisualizer):
    def __init__(self, thickness: int = 1, color: tuple = (0, 255, 0)):
        self._thick = thickness
        self._color = color

    def process_img(self, image, mask) -> np.ndarray:
        cntrs, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return cv2.drawContours(image, cntrs, -1, self._color, self._thick)


class MulticlassColormapVisualizer(ColormapVisualizer):
    def __init__(self, main_class: int, proportions: [float, float], colormap=cv2.COLORMAP_JET, other_colors: [] = None):
        super().__init__(proportions, colormap)

        self._main_class = main_class
        self._other_colors = other_colors

    def process_img(self, image, mask) -> np.ndarray:
        main_target = mask[:, :, self._main_class]
        other_classes = np.delete(mask, self._main_class, 2)
        img = super().process_img(image, main_target)

        if self._other_colors is None:
            self._other_colors = np.array([np.linspace(127, 0, num=other_classes.shape[0], dtype=np.uint8),
                                           np.linspace(255, 127, num=other_classes.shape[0], dtype=np.uint8),
                                           np.linspace(127, 255, num=other_classes.shape[0], dtype=np.uint8)], dtype=np.uint8)
        for i in range(other_classes.shape[2]):
            cls = other_classes[:, :, i]
            img[:, :, 0][cls > 0] = self._other_colors[0][i]
            img[:, :, 1][cls > 0] = self._other_colors[1][i]
            img[:, :, 2][cls > 0] = self._other_colors[2][i]
        return img


class DetectionVisualiser(AbstractVisualizer):
    def process_img(self, image: np.ndarray, target: any) -> np.ndarray:
        res = image.copy()
        for bbox in target:
            res = cv2.rectangle(res, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 0, 0), 2)
        return res
