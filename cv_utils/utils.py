import numpy as np

__all__ = ['mask2rle', 'rle2mask']


def mask2rle(img: np.ndarray):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.copy().T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(rle: [int], width: int, height: int):
    mask = np.zeros(width * height)
    array = np.asarray(rle)
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position + lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)


def generate_folds_names(folds_num: int) -> list:
    fold_indices = list(range(folds_num))
    folds = []
    for _ in range(folds_num):
        fold_indices = fold_indices[-1:] + fold_indices[:-1]
        folds.append({'train': ['fold_{}.npy'.format(i) for i in fold_indices[1:]], 'val': 'fold_{}.npy'.format(fold_indices[0])})
    return folds
