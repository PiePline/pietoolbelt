import numpy as np
import cv2

__all__ = ['MasksComposer']


class MasksComposer:
    def __init__(self, target_shape: [], dtype: np.typename = np.uint8):
        self._masks = {}
        self._mask_shape = target_shape
        self._type = dtype

        self._borders_as_class = False
        self._borders_between_classes = None
        self._dilate_masks_kernel = None

    def add_borders_as_class(self, between_classes: [] = None, dilate_masks_kernel: np.ndarray = np.ones((2, 2), dtype=np.uint8)) -> 'MasksComposer':
        self._borders_as_class = True
        self._borders_between_classes = between_classes
        self._dilate_masks_kernel = dilate_masks_kernel

    def _calc_border_between_masks(self, mask1, mask2):
        borders = np.zeros_like(mask1)
        mask1_intern = cv2.dilate(mask1, self._dilate_masks_kernel)
        mask2_intern = cv2.dilate(mask2, self._dilate_masks_kernel)

        add = mask1_intern + mask2_intern
        borders[add > 1] = 1
        return borders

    def add_mask(self, mask: np.ndarray, cls, offset: np.ndarray = None):
        if cls not in self._masks:
            self._masks[cls] = np.zeros(self._mask_shape, dtype=self._type)

        if self._borders_as_class and cls in self._borders_between_classes:
            prev_borders = None
            if offset is not None:
                if isinstance(self._masks[cls], dict):
                    target_mask = np.zeros_like(self._masks[cls]['mask'])
                    prev_borders = self._masks[cls]['borders']
                else:
                    target_mask = np.zeros_like(self._masks[cls])
                target_mask[offset[0]: offset[0] + mask.shape[0], offset[1]: offset[1] + mask.shape[1]] = mask
            else:
                target_mask = mask

            origin_mask = self._masks[cls] if not isinstance(self._masks[cls], dict) else self._masks[cls]['mask']
            borders = self._calc_border_between_masks(origin_mask, target_mask)

            if prev_borders is not None:
                borders += prev_borders

            origin_mask += target_mask
            self._masks[cls] = {'mask': np.clip(origin_mask, 0, 1), 'borders': np.clip(borders, 0, 1)}
        else:
            if offset is not None:
                self._masks[cls][offset[0]: offset[0] + mask.shape[0], offset[1]: offset[1] + mask.shape[1]] += mask
            else:
                self._masks[cls] += mask

    def compose(self) -> np.ndarray:
        res = None
        for cls, mask in self._masks.items():
            if res is None:
                if isinstance(mask, dict):
                    res = np.stack((mask['mask'], mask['borders']), axis=2)
                else:
                    res = mask
            else:
                if isinstance(mask, dict):
                    res = np.stack((res, mask['mask'], mask['borders']), axis=2)
                else:
                    res = np.stack((res, mask), axis=0)
        return res
