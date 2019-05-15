import os

import cv2
from neural_pipeline import AbstractDataset

from cv_utils.datasets.common import get_root_by_env

__all__ = ['PicsartDataset']


class PicsartDataset(AbstractDataset):
    def __init__(self, images_pathes: []):
        base_dir = get_root_by_env('PICSART_DATASET')

        images_dir = os.path.join(base_dir, 'train')
        masks_dir = os.path.join(base_dir, 'train_mask')
        images_pathes = sorted(images_pathes, key=lambda p: int(os.path.splitext(p)[0]))
        self._image_pathes = []
        for p in images_pathes:
            name = os.path.splitext(p)[0]
            mask_img = os.path.join(masks_dir, name + '.png')
            if os.path.exists(mask_img):
                path = {'data': os.path.join(images_dir, p), 'target': mask_img}
                self._image_pathes.append(path)

    def __len__(self):
        return len(self._image_pathes)

    def __getitem__(self, item):
        img = cv2.cvtColor(cv2.imread(self._image_pathes[item]['data']), cv2.COLOR_BGR2RGB)
        return {'data': img, 'target': cv2.imread(self._image_pathes[item]['target'], cv2.IMREAD_UNCHANGED) / 255}
