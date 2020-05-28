from typing import List

import cv2
import torch
from albumentations import BasicTransform
from torch.nn import Module

import numpy as np

from pietoolbelt.tta import AbstractTTA
from pietoolbelt.viz import ColormapVisualizer


class SegmentationInference:
    def __init__(self, model: Module):
        self._model = model

        self._transform = None
        self._target_transform = None
        self._tta = None

        self._threshold = 0.5

        self._vis = ColormapVisualizer([0.5, 0.5])

        self._device = None

    def set_device(self, device: str) -> 'SegmentationInference':
        self._model = self._model.to(device)
        self._device = device
        return self

    def set_data_transform(self, transform: BasicTransform) -> 'SegmentationInference':
        self._transform = transform
        return self

    def set_target_transform(self, transform: BasicTransform) -> 'SegmentationInference':
        self._target_transform = transform
        return self

    def set_tta(self, tta: List[AbstractTTA]) -> 'SegmentationInference':
        self._tta = tta
        return self

    def _process_imag(self, image) -> np.ndarray:
        data = np.swapaxes(image, 0, -1).astype(np.float32) / 128 - 1
        data = torch.from_numpy(np.expand_dims(data, axis=0))

        if self._device is not None:
            data = data.to(self._device)

        res = self._model(data).detach().cpu().numpy()
        res = np.squeeze(res)

        if self._target_transform is not None:
            res = self._target_transform(image=res)['image']

        return res

    def set_threshold(self, thresh: float) -> 'SegmentationInference':
        self._threshold = thresh
        return self

    def run_image(self, image: np.ndarray) -> [np.ndarray, np.ndarray]:
        if self._transform is not None:
            image = self._transform(image=image)['image']

        res = self._process_imag(image)

        if self._tta is not None:
            results = [res]
            for tta in self._tta:
                cur_image = tta.preprocess(image)
                res = self._process_imag(cur_image)
                results.append(tta.postprocess(res))
            res = np.mean(results, axis=0)

        res[res < self._threshold] = 0
        res = 255 * (res + res.min()) / (res.max() - res.min())
        res = res.astype(np.uint8)
        res[res > 0] = 255

        return image, res

    def vis_result(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self._vis.process_img(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), mask)

    def run_webcam(self, title: str = 'Inference', device_id: int = 0):
        cap = cv2.VideoCapture(device_id)
        cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL)

        while cv2.waitKey(1) & 0xFF != ord('q'):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame, res = self.run_image(frame)
            image = self.vis_result(frame, res)
            cv2.imshow(title, image)

        cap.release()
        cv2.destroyAllWindows()
