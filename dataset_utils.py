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
                    res = np.stack((mask['mask'], mask['borders']), axis=0)
                else:
                    res = mask
            else:
                if isinstance(mask, dict):
                    res = np.stack((res, mask['mask'], mask['borders']), axis=0)
                else:
                    res = np.stack((res, mask), axis=0)
        return res


class SuperviselyDataset(AbstractDataset):
    def __init__(self, path: str, include_not_marked_people: bool = False, include_neutral_objects: bool = False):
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
                obj_mask = cv2.drawContours(mask.copy(), [pts], -1, 1, cv2.FILLED)
                origin = pts.min(axis=0)

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


class EmptyClassesAdd:
    def __init__(self, dataset, target_classes_num: int, exists_class_idx: int):
        if target_classes_num <= exists_class_idx:
            raise Exception("Target classes number ({}) can't be less or equal than exists class index ({})".format(target_classes_num, exists_class_idx))

        self._dataset = dataset
        self._target_classes_num = target_classes_num
        self._exists_class_idx = exists_class_idx

    def __getitem__(self, item):
        cur_res = self._dataset[item]
        res = {'data': cur_res['data']}
        target_shape = cur_res['target'].shape
        if target_shape[0] > self._target_classes_num:
            raise RuntimeError("Dataset produce target with shape, that's first dimension greater than target classes num."
                               "Target shape from dataset: {}, target classes num: {}".format(target_shape, self._target_classes_num))
        if len(target_shape) > 3:
            raise RuntimeError("Dataset produce target with channels number more than 3."
                               "Target shape from dataset: {}".format(target_shape))
        target_shape = (self._target_classes_num, target_shape[0], target_shape[1])
        target = np.zeros_like(target_shape)
        target[self._exists_class_idx, :, :] = cur_res['shape']
        res['target'] = target
        return res


class AugmentedDataset:
    def __init__(self, dataset):
        self._dataset = dataset
        self._augmentations = {}

    def add_aug(self, aug: callable, identificator) -> 'AugmentedDataset':
        self._augmentations[identificator] = aug
        return self

    def __getitem__(self, item):
        res = self._dataset[item]
        for k, v in res.items() if isinstance(res, dict) else enumerate(res):
            res[k] = self._augmentations[k](v)
        return res

class SegmentationVisualizer(metaclass=ABCMeta):
    @abstractmethod
    def process_img(self, image, mask) -> np.ndarray:
        """
        Combine image and mask into rgb image to visualize

        :param image: image
        :param mask: mask
        :return: new image
        """


class ColormapVisualizer(SegmentationVisualizer):
    def __init__(self, proportions: [float, float], colormap=cv2.COLORMAP_JET):
        self._proportions = proportions
        self._colormap = colormap

    def process_img(self, image, mask) -> np.ndarray:
        heatmap_img = cv2.applyColorMap(mask, self._colormap)
        return cv2.addWeighted(heatmap_img, self._proportions[1], image, self._proportions[0], 0)


class MulticlassColormapVisualizer(ColormapVisualizer):
    def __init__(self, main_class: int, proportions: [float, float], colormap=cv2.COLORMAP_JET, other_colors: []=None):
        super().__init__(proportions, colormap)

        self._main_class = main_class
        self._other_colors = other_colors

    def process_img(self, image, mask) -> np.ndarray:
        main_target = mask[self._main_class, :, :]
        other_classes = np.delete(mask, self._main_class, 0)
        img = super().process_img(image, main_target)

        if self._other_colors is None:
            self._other_colors = np.array([np.linspace(127, 0, num=other_classes.shape[0], dtype=np.uint8),
                                           np.linspace(255, 127, num=other_classes.shape[0], dtype=np.uint8),
                                           np.linspace(127, 255, num=other_classes.shape[0], dtype=np.uint8)], dtype=np.uint8)
        for i, cls in enumerate(other_classes):
            img[:, :, 0][cls > 0] = self._other_colors[0][i]
            img[:, :, 1][cls > 0] = self._other_colors[1][i]
            img[:, :, 2][cls > 0] = self._other_colors[2][i]
        return img
