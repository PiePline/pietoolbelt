import numpy as np
from cv_utils.mask_composer import MasksComposer
from neural_pipeline import AbstractDataset

__all__ = ['EmptyClassesAdd', 'AugmentedDataset']


class EmptyClassesAdd:
    def __init__(self, dataset, target_classes_num: int, exists_class_idx: int):
        if target_classes_num <= exists_class_idx:
            raise Exception("Target classes number ({}) can't be less or equal than exists class index ({})".format(target_classes_num,
                                                                                                                    exists_class_idx))

        self._dataset = dataset
        self._target_classes_num = target_classes_num
        self._exists_class_idx = exists_class_idx

    def __getitem__(self, item):
        cur_res = self._dataset[item]
        res = {'data': cur_res['data']}
        target_shape = cur_res['target'].shape
        if len(target_shape) > 3:
            raise RuntimeError("Dataset produce target with channels number more than 3."
                               "Target shape from dataset: {}".format(target_shape))
        elif len(target_shape) == 3 and target_shape[2] > self._target_classes_num:
            raise RuntimeError("Dataset produce target with shape, that's first dimension greater than target classes num."
                               "Target shape from dataset: {}, target classes num: {}".format(target_shape, self._target_classes_num))

        target_shape = (target_shape[0], target_shape[1], self._target_classes_num)
        target = np.zeros(target_shape, dtype=np.uint8)
        target[:, :, self._exists_class_idx] = cur_res['target']
        res['target'] = target
        return res

    def __len__(self):
        return len(self._dataset)


class AugmentedDataset:
    def __init__(self, dataset):
        self._dataset = dataset
        self._augs = {}
        self._augs_for_whole = []

    def add_aug(self, aug: callable, identificator=None) -> 'AugmentedDataset':
        if identificator is None:
            self._augs_for_whole.append(aug)
        else:
            self._augs[identificator] = aug
        return self

    def __getitem__(self, item):
        res = self._dataset[item]
        for k, v in res.items() if isinstance(res, dict) else enumerate(res):
            if k in self._augs:
                res[k] = self._augs[k](v)
        for aug in self._augs_for_whole:
            res = aug(res)
        return res

    def __len__(self):
        return len(self._dataset)


class MulticlassSegmentationDataset(AbstractDataset):
    def __init__(self, dataset: AbstractDataset, target_key: str = 'target'):
        self._dataset = dataset
        self._target_key = target_key

        self._border_thickness = None
        self._border_cls_pos = None

    def enable_border(self, thickness: int, border_cls_position: int = 1) -> 'MulticlassSegmentationDataset':
        self._border_thickness = thickness
        self._border_cls_pos = border_cls_position
        return self

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item: int):
        res = self._dataset[item]

        target = res[self._target_key]
        target_shape = (target['size']['height'], target['size']['width'])
        composer = MasksComposer(target_shape)

        if self._border_thickness is None:
            composer.add_borders_as_class(between_classes=[self._border_cls_pos])
        else:
            composer.add_borders_as_class(between_classes=[self._border_cls_pos],
                                          dilate_masks_kernel=np.ones((self._border_thickness, self._border_thickness),
                                                                      dtype=np.uint8))

        for obj in target['masks']:
            composer.add_mask(obj[0], 0, offset=obj[1])

        return composer.compose()
