from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
from albumentations import Compose, OneOf, HorizontalFlip, GaussNoise, RandomBrightnessContrast, RandomGamma, Rotate, \
    ImageCompression, CLAHE, Downscale, ISONoise, MotionBlur

__all__ = ['to_quad', 'BaseAugmentations']


def to_quad(force_apply=False, **kwargs):
    image = kwargs['image']
    max_size, min_size = np.max(image.shape), np.min([image.shape[0], image.shape[1]])
    image_tmp = np.ones((max_size, max_size, image.shape[2]), dtype=np.uint8)
    pos = (max_size - min_size) // 2

    if image.shape[0] > image.shape[1]:
        image_tmp[:, pos: pos + min_size, :] = image
    else:
        image_tmp[pos: pos + min_size, :, :] = image

    if 'mask' in kwargs:
        mask_tmp = np.zeros((max_size, max_size, 3), dtype=np.uint8)

        if image.shape[0] > image.shape[1]:
            mask_tmp[:, pos: pos + min_size, :] = kwargs['mask']
        else:
            mask_tmp[pos: pos + min_size, :, :] = kwargs['mask']
        return {'image': image_tmp, 'mask': mask_tmp}

    # if 'bboxes' in kwargs:


    return {'image': image_tmp}


class BaseAugmentations(metaclass=ABCMeta):
    def __init__(self, is_train: bool, to_pytorch: bool, preprocess: callable):
        if is_train:
            self._aug = Compose([
                preprocess,
                OneOf([
                    Compose([
                        HorizontalFlip(p=0.5),
                        GaussNoise(p=0.5),
                        OneOf([
                            RandomBrightnessContrast(),
                            RandomGamma(),
                        ], p=0.5),
                        Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT),
                        ImageCompression(),
                        CLAHE(),
                        Downscale(scale_min=0.2, scale_max=0.9, p=0.5),
                        ISONoise(p=0.5),
                        MotionBlur(p=0.5)
                    ]),
                    HorizontalFlip(p=0.5)
                ])
            ], p=1)
        else:
            self._aug = preprocess

        self._need_to_pytorch = to_pytorch

    @abstractmethod
    def augmentation(self, data: dict) -> dict:
        """
        Perform data augmentation

        Args:
            data: dict of data with keys {'data', 'target'}

        Returns:
            dict of augmented data with structure as input `data`
        """