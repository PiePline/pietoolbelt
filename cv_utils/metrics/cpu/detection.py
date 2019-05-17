import numpy as np

from cv_utils.box_utils import calc_boxes_areas


def _compute_boxes_iou(box: np.ndarray or [], boxes: np.ndarray or [], box_area: float, boxes_area: float):
    """
    Calculates IoU of the given box with the array of the given boxes.

    Args:
        box: 1D vector [y1, x1, y2, x2]
        boxes: [boxes_count, (y1, x1, y2, x2)]
        box_area: float. the area of 'box'
        boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    y1 = np.maximum(box[0], boxes[:, :, 0])
    x1 = np.maximum(box[1], boxes[:, :, 1])
    x2 = np.minimum(box[2], boxes[:, :, 2])
    y2 = np.minimum(box[3], boxes[:, :, 3])
    intersection = np.maximum(x2 - x1, 1) * np.maximum(y2 - y1, 1)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def f_beta_score(pred: np.ndarray, target: np.ndarray, beta: int, thresholds: [float]):
    """
    Calculate F-Beta score.

    Args:
        pred (np.ndarray): predicted bboxes of shape [B, N, 4]
        target (np.ndarray): target bboxes of shape [B, N, 4]
        beta (int): value of Beta coefficient
        There is N - number of instance masks

    Returns:
        np.ndarray: array with values of F-Beta score. Array shape: [B]
    """
    pred_areas = calc_boxes_areas(pred)  # [B, N], [B, N]
    target_areas = calc_boxes_areas(target)  # [B, N], [B, N]

    ious = []
    for batch_idx in enumerate(pred.shape[0]):
        for instance_idx in enumerate(pred.shape[1]):
            _compute_boxes_iou(pred[batch_idx][instance_idx], target[batch_idx], pred_areas[batch_idx][instance_idx], target_areas[batch_idx])

    raise NotImplementedError()
