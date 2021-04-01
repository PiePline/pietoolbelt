import numpy as np
import torch

from .common import BaseAugmentations

__all__ = ['SegmentationAugmentations']


class SegmentationAugmentations(BaseAugmentations):
    def __init__(self, is_train: bool, to_pytorch: bool, preprocess: callable):
        super().__init__(is_train, to_pytorch, preprocess)

    def augmentation(self, data: dict) -> dict:
        augmented = self._aug(image=data['data'], mask=data['target'] / (data['target'].max() + 1e-7))

        img, mask = augmented['image'], augmented['mask']
        if self._need_to_pytorch:
            img, mask = self.img_to_pytorch(img), self.mask_to_pytorch(mask)

        return {'data': img, 'target': mask}

    @staticmethod
    def img_to_pytorch(image):
        return torch.from_numpy(np.expand_dims(np.moveaxis(image, -1, 0).astype(np.float32) / 128 - 1, axis=0))

    @staticmethod
    def mask_to_pytorch(mask):
        return torch.from_numpy(np.expand_dims(mask.astype(np.float32), axis=0))
