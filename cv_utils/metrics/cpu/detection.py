import numpy as np

from cv_utils.box_utils import calc_boxes_areas


def _compute_boxes_iou(box: np.ndarray or [], boxes: np.ndarray or [], box_area: float, boxes_area: [float]) -> np.ndarray:
    """
    Calculates IoU of the given box with the array of the given boxes.

    Args:
        box: 1D vector [y1, x1, y2, x2]
        boxes: [boxes_count, (y1, x1, y2, x2)]
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
    intersection[xmin > xmax] = 0
    intersection[ymin > ymax] = 0
    intersection[intersection < 0] = 0
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def calc_tp_fp_fn(pred: np.ndarray, target: np.ndarray, threshold: float):
    """
    Calculate true positives, false positives and false negatives number for predicted and target boxes

    Args:
        pred: Array of predicted boxes with shape [N, 4]
        target: Array of ground truth boxes with shape [N, 4]
        threshold: the threshold for iou metric

    Returns:
        Return list of [tp, fp, fn]
    """
    pred_areas = calc_boxes_areas(pred)  # [N], [N]
    target_areas = calc_boxes_areas(target)  # [N], [N]
    ious = []
    for instance_idx in range(pred.shape[0]):
        ious.append(_compute_boxes_iou(pred[instance_idx], target, pred_areas[instance_idx], target_areas))

    matches_matrix = np.array(ious)
    matches_matrix[matches_matrix < threshold] = 0
    tp = matches_matrix[matches_matrix > 0].shape[0]
    fn = target.shape[0] - tp
    fp = pred.shape[0] - tp

    return tp, fn, fp


def f_beta_score(pred: np.ndarray, target: np.ndarray, beta: int, thresholds: [float]):
    """
    Calculate F-Beta score.

    Args:
        pred (np.ndarray): predicted bboxes of shape [B, N, 4]
        target (np.ndarray): target bboxes of shape [B, N, 4]
        beta (int): value of Beta coefficient
        thresholds ([float]): list of thresholds
        There is N - number of instance masks

    Returns:
        np.ndarray: array with values of F-Beta score. Array shape: [B]
    """
    beta_squared = beta ** 2
    res = []
    for batch_idx in range(pred.shape[0]):
        batch_results = []
        for thresh in thresholds:
            tp, fp, fn = calc_tp_fp_fn(pred[batch_idx], target[batch_idx], thresh)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            batch_results.append((beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + 1e-7))

        res.append(np.mean(batch_results))
    return np.array(res)
