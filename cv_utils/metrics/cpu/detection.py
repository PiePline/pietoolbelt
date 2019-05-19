import numpy as np

from cv_utils.box_utils import calc_boxes_areas


def _compute_boxes_iou(box: np.ndarray or [], boxes: np.ndarray or [], box_area: float, boxes_area: [float]) -> np.ndarray:
    """
    Calculates IoU of the given box with the array of the given boxes.

    Args:
        box: 1D vector [y1, x1, y2, x2]
        boxes: [batch, boxes_count, (y1, x1, y2, x2)]
        box_area: float. the area of 'box'
        boxes_area: array of boxes areas with shape [batch, boxes_count].

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.

    Returns:
        array of iou in shape []
    """
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
    intersection = (xmax - xmin) * (ymax - ymin)
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
    ious = []
    for batch_idx in range(pred.shape[0]):
        pred_areas = calc_boxes_areas(pred[batch_idx])  # [N], [N]
        target_areas = calc_boxes_areas(target[batch_idx])  # [N], [N]
        for instance_idx in range(pred.shape[1]):
            ious.append(_compute_boxes_iou(pred[batch_idx][instance_idx], target[batch_idx], pred_areas[instance_idx], target_areas))

        for thresh in thresholds:
            pass
    raise NotImplementedError()
