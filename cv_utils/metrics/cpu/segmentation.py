import numpy as np

__all__ = ['dice', 'jaccard', 'multiclass_dice', 'multiclass_jaccard', 'pixelwise_f_beta_score']


def _split_masks_by_classes(pred: np.ndarray, target: np.ndarray) -> []:
    """
    Split masks by classes

    Args:
        pred (np.ndarray): predicted masks of shape [B, C, H, W]
        target (np.ndarray): target masks of shape [B, C, H, W]

    Returns:
        List: list of masks pairs [pred, target], splitted by channels. List shape: [C, 2, B, H, W]
    """
    preds = np.split(pred, pred.shape[1], axis=1)
    targets = np.split(target, target.shape[1], axis=1)

    return list(zip(preds, targets))


def dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Calculate Dice coefficient

    Args:
        pred (np.ndarray): predicted masks of shape [B, H, W]
        target (np.ndarray): target masks of shape [B, H, W]
        eps (float): smooth value

    Returns:
        np.ndarray: array with values of Dice coefficient. Array shape: [B]
    """
    preds_inner = np.reshape(pred, (pred.shape[0], pred.size // pred.shape[0]))
    trues_inner = np.reshape(target, (target.shape[0], target.size // target.shape[0]))
    intersection = (preds_inner * trues_inner).sum(1)

    res = (2. * intersection + eps) / ((preds_inner + trues_inner).sum(1) + eps)

    return res


def jaccard(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Calculate Jaccard coefficient

    Args:
        pred (np.ndarray): predicted masks of shape [B, H, W]
        target (np.ndarray): target masks of shape [B, H, W]
        eps (float): smooth value

    Returns:
        np.ndarray: array with values of Jaccard coefficient. Array shape: [B]
    """
    preds_inner = np.reshape(pred, (pred.shape[0], pred.size // pred.shape[0]))
    trues_inner = np.reshape(target, (target.shape[0], target.size // target.shape[0]))
    intersection = (preds_inner * trues_inner).sum(1)
    scores = (intersection + eps) / ((preds_inner + trues_inner).sum(1) - intersection + eps)

    return scores


def _multiclass_metric(func: callable, pred, target, eps: float = 1e-7):
    res = np.zeros((pred.shape[1], pred.shape[0]))
    for i, [p, t] in enumerate(_split_masks_by_classes(pred, target)):
        res[i] = func(p, t, eps)
    return res


def multiclass_dice(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7):
    """
    Calculate Dice coefficient for multiclass case

    Args:
        pred (np.ndarray): predicted masks of shape [B, C, H, W]
        target (np.ndarray): target masks of shape [B, C, H, W]
        eps (float): smooth value

    Returns:
        np.ndarray: Array with values of Dice coefficient. Array shape: [C, B]
    """
    return _multiclass_metric(dice, pred, target, eps)


def multiclass_jaccard(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7):
    """
    Calculate Jaccard coefficient for multiclass case

    Args:
        pred (np.ndarray): predicted masks of shape [B, C, H, W]
        target (np.ndarray): target masks of shape [B, C, H, W]
        eps (float): smooth value

    Returns:
        np.ndarray: Array with values of Jaccard coefficient. Array shape: [C, B]
    """
    return _multiclass_metric(jaccard, pred, target, eps)


def f_beta_score(pred: np.ndarray, target: np.ndarray, beta: int):
    """
    Calculate F-Beta score.

    Args:
        pred (np.ndarray): predicted masks of shape [B, H, W]
        target (np.ndarray): target masks of shape [B, N, H, W]. There is N - number of instance masks
        beta (int): value of Beta coefficient

    Returns:
        np.ndarray: array with values of F-Beta score. Array shape: [B]
    """
    raise NotImplementedError()


def pixelwise_f_beta_score(pred: np.ndarray, target: np.ndarray, beta: int, threshold_shift: float = 0):
    y_pred_bin = np.round(pred + threshold_shift)

    tp = np.sum(np.round(target * y_pred_bin)) + 1e-7
    fp = np.sum(np.round(np.clip(y_pred_bin - target, 0, 1)))
    fn = np.sum(np.round(np.clip(target - pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + 1e-7)
