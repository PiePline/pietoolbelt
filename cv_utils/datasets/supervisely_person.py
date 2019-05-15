import base64
import json
import os
import zlib
import numpy as np
import cv2
from neural_pipeline import AbstractDataset

from cv_utils.datasets.common import get_root_by_env, MasksComposer

__all__ = ['SuperviselyPersonDataset']


class SuperviselyPersonDataset(AbstractDataset):
    def __init__(self, include_not_marked_people: bool = False, include_neutral_objects: bool = False):
        path = get_root_by_env('SUPERVISELY_DATASET')

        items = {}
        for root, path, files in os.walk(path):
            for file in files:
                name, ext = os.path.splitext(file)

                if ext == '.json':
                    item_type = 'target'
                elif ext == '.png' or ext == '.jpg':
                    item_type = 'data'
                else:
                    continue

                if name in items:
                    items[name][item_type] = os.path.join(root, file)
                else:
                    items[name] = {item_type: os.path.join(root, file)}

        self._items = []
        for item, data in items.items():
            if 'data' in data and 'target' in data:
                self._items.append(data)

        self._items = self._filter_items(self._items, include_not_marked_people, include_neutral_objects)
        self._use_border_as_class = False
        self._border_thikness = None

    def use_border_as_class(self, border_thikness: int = None):
        self._use_border_as_class = True
        self._border_thikness = border_thikness

    @staticmethod
    def _filter_items(items, include_not_marked_people: bool, include_neutral_objects: bool) -> []:
        res = []
        for item in items:
            with open(item['target'], 'r') as file:
                target = json.load(file)

            if not include_not_marked_people and 'not-marked-people' in target['tags']:
                continue

            if not include_neutral_objects:
                res_objects = []
                for obj in target['objects']:
                    if obj['classTitle'] != 'neutral':
                        res_objects.append(obj)
                target['objects'] = res_objects

            res.append({'data': item['data'], 'target': target})

        return res

    def _process_target(self, target: {}):
        def object_to_mask(obj):
            obj_mask, origin = None, None

            if obj['bitmap'] is not None:
                z = zlib.decompress(base64.b64decode(obj['bitmap']['data']))
                n = np.fromstring(z, np.uint8)

                origin = np.array([obj['bitmap']['origin'][1], obj['bitmap']['origin'][0]], dtype=np.uint16)
                obj_mask = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)[:, :, 3].astype(np.uint8)
                obj_mask[obj_mask > 0] = 1

            elif len(obj['points']['interior']) + len(obj['points']['exterior']) > 0:
                pts = np.array(obj['points']['exterior'], dtype=np.int)
                origin = pts.min(axis=0)
                shape = pts.max(axis=0) - origin
                obj_mask = cv2.drawContours(np.zeros((shape[1], shape[0]), dtype=np.uint8), [pts - origin], -1, 1, cv2.FILLED)
                origin = np.array([origin[1], origin[0]], dtype=np.int)

            return obj_mask, origin

        target_shape = (target['size']['height'], target['size']['width'])
        composer = MasksComposer(target_shape)

        if self._use_border_as_class:
            if self._border_thikness is None:
                composer.add_borders_as_class(between_classes=[0])
            else:
                composer.add_borders_as_class(between_classes=[0], dilate_masks_kernel=np.ones((self._border_thikness, self._border_thikness), dtype=np.uint8))

        for obj in target['objects']:
            mask, origin = object_to_mask(obj)
            composer.add_mask(mask, 0, offset=origin)

        return composer.compose()

    def __len__(self):
        return len(self._items)

    def __getitem__(self, item):
        return {'data': cv2.imread(self._items[item]['data']), 'target': self._process_target(self._items[item]['target'])}
