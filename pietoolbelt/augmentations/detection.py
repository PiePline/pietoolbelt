import numpy as np
import torch
from albumentations import Compose, BboxParams

from .common import BaseAugmentations

__all__ = ['DetectionAugmentations']


class DetectionAugmentations(BaseAugmentations):
    def __init__(self, is_train: bool, to_pytorch: bool, preprocess: callable):
        super().__init__(is_train, to_pytorch, preprocess)
        self._aug = Compose([self._aug], bbox_params=BboxParams(format='coco', label_fields=['category_ids']))

    def augmentation(self, data: dict) -> dict:
        bboxes = data['target']

        augmented = self._aug(image=data['data'], bboxes=bboxes, category_ids=np.zeros((len(bboxes))))
        img, bboxes = augmented['image'], np.array(augmented['bboxes'], dtype=np.float32)

        if self._need_to_pytorch:
            img, mask = self.img_to_pytorch(img), self.bbox_to_pytorch(bboxes)

        return {'data': img, 'target': bboxes}

    @staticmethod
    def img_to_pytorch(image):
        return torch.from_numpy(np.expand_dims(np.moveaxis(image, -1, 0).astype(np.float32) / 128 - 1, axis=0))

    @staticmethod
    def bbox_to_pytorch(bbox):
        return torch.from_numpy(np.expand_dims(bbox.astype(np.float32), axis=0))
