import numpy as np


def split_masks_by_classes(pred: np.ndarray, target: np.ndarray):
    preds = np.split(pred, 1, dim=1)
    targets = np.split(target, 1, dim=1)

    return list(zip(preds, targets))


def dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7):
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)

    intersection = np.sum(iflat * tflat)

    res = (2. * intersection + eps) / (np.sum(iflat) + np.sum(tflat) + eps)

    return res


def jaccard(preds: np.ndarray, trues: np.ndarray, eps: float = 1e-7):
    preds_inner = preds.cpu().data.numpy().copy()
    trues_inner = trues.cpu().data.numpy().copy()

    preds_inner = np.reshape(preds_inner, (preds_inner.shape[0], preds_inner.size // preds_inner.shape[0]))
    trues_inner = np.reshape(trues_inner, (trues_inner.shape[0], trues_inner.size // trues_inner.shape[0]))
    intersection = (preds_inner * trues_inner).sum(1)
    scores = (intersection + eps) / ((preds_inner + trues_inner).sum(1) - intersection + eps)

    return scores


def multiclass_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7):
    res, num = 0, 0
    for p, t in split_masks_by_classes(pred, target):
        res += dice(p, t, eps)
        num += 1
    return res / num


def multiclass_jaccard(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7):
    res, num = 0, 0
    for p, t in split_masks_by_classes(pred, target):
        res += jaccard(p, t, eps)
        num += 1
    return res / num
