from typing import Tuple

import numpy as np

__all__ = ['mask2rle', 'rle2mask', 'put_to_dict', 'get_from_dict']


def mask2rle(img: np.ndarray):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.copy().flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(rle: [int], shape: Tuple[int, int]):
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    array = np.asarray(rle)
    starts = array[0::2]
    lengths = array[1::2]

    for index, start in enumerate(starts):
        mask[start:start + lengths[index]] = 255

    return mask.reshape(shape)


def generate_folds_names(folds_num: int) -> list:
    fold_indices = list(range(folds_num))
    folds = []
    for _ in range(folds_num):
        fold_indices = fold_indices[-1:] + fold_indices[:-1]
        folds.append({'train': ['fold_{}.npy'.format(i) for i in fold_indices[1:]], 'val': 'fold_{}.npy'.format(fold_indices[0])})
    return folds


def put_to_dict(dict_obj: dict, path: list, obj: any) -> dict:
    cur = dict_obj
    for p in path:
        if p not in cur:
            cur[p] = obj
        cur = cur[p]
    return dict_obj


def get_from_dict(dict_obj: dict, path: list) -> any:
    res = dict_obj
    for p in path:
        res = res[p]
    return res
