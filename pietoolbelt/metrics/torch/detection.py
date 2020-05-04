import torch
from piepline import AbstractMetric
from torch import Tensor

__all__ = ['calc_tp_fp_fn', 'f_beta_score']


def _calc_boxes_areas(boxes: Tensor):
    """
    Calculate areas of array of boxes

    :param boxes: array of boxes with shape [N, 4]
    :return: array of boxes area with shape [N]
    """
    xx, yy = boxes[:, 0::2], boxes[:, 1::2]  # [N, 2], [N, 2]
    boxes_x_min, boxes_x_max = xx[:, 0], xx[:, 1]  # [N], [N]
    boxes_y_min, boxes_y_max = yy[:, 0], yy[:, 1]  # [N], [N]
    return (boxes_x_max - boxes_x_min) * (boxes_y_max - boxes_y_min)  # [N], [N]


def _compute_boxes_iou(box: Tensor, boxes: Tensor, box_area: float, boxes_area: [float]) -> Tensor:
    """
    Calculates IoU of the given box with the array of the given boxes.

    Args:
        box: 1D vector [y1, x1, y2, x2]
        boxes: [N, (y1, x1, y2, x2)]
        box_area: float. the area of 'box'
        boxes_area: array of boxes areas with shape [batch, boxes_count].

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.

    Returns:
        Tensor of iou with size [N]
    """
    xmin = torch.max(box[0], boxes[:, 0])
    ymin = torch.max(box[1], boxes[:, 1])
    xmax = torch.min(box[2], boxes[:, 2])
    ymax = torch.min(box[3], boxes[:, 3])
    intersection = (xmax - xmin) * (ymax - ymin)
    intersection[xmin > xmax] = 0
    intersection[ymin > ymax] = 0
    intersection[intersection < 0] = 0
    union = box_area + boxes_area - intersection
    iou = intersection / union
    return iou


def calc_tp_fp_fn(pred: Tensor, target: Tensor, threshold: float) -> tuple:
    """
    Calculate true positives, false positives and false negatives number for predicted and target boxes

    Args:
        pred: Array of predicted boxes with shape [N, 4]
        target: Array of ground truth boxes with shape [N, 4]
        threshold: the threshold for iou metric

    Returns:
        Return list of [tp, fp, fn]
    """
    pred_areas = _calc_boxes_areas(pred)  # [N], [N]
    target_areas = _calc_boxes_areas(target)  # [N], [N]
    matches_matrix = torch.zeros((pred.size(0), target.size(0)))
    for instance_idx in range(pred.size(0)):
        matches_matrix[instance_idx] = _compute_boxes_iou(pred[instance_idx], target, pred_areas[instance_idx], target_areas)

    matches_matrix[matches_matrix < threshold] = 0
    tp = matches_matrix[matches_matrix > 0].shape[0]
    fn = target.shape[0] - tp
    fp = pred.shape[0] - tp

    return tp, fn, fp


def f_beta_score(pred: Tensor, target: Tensor, beta: int, thresholds: [float]) -> Tensor:
    """
    Calculate F-Beta score.

    Args:
        pred (Tensor): predicted bboxes of shape [B, N, 4]
        target (Tensor): target bboxes of shape [B, N, 4]
        beta (int): value of Beta coefficient
        thresholds ([float]): list of thresholds
        There is N - number of instance masks

    Returns:
        Tensor: array with values of F-Beta score. Array shape: [B]
    """
    beta_squared = beta ** 2
    res = []
    for batch_idx in range(pred.shape[0]):
        batch_results = torch.zeros(len(thresholds))
        for i, thresh in enumerate(thresholds):
            tp, fp, fn = calc_tp_fp_fn(pred[batch_idx], target[batch_idx], thresh)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            batch_results[i] = (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + 1e-7)

        res.append(torch.mean(batch_results))
    return torch.FloatTensor(res)


class FBetaMetric(AbstractMetric):
    def __init__(self, beta: int, thresholds: [float]):
        super().__init__('f_beta')
        self._beta = beta
        self._thresholds = thresholds

    def calc(self, output: Tensor, target: Tensor) -> Tensor or float:
        return f_beta_score(output.data.cpu().numpy(), target.data.cpu().numpy(), self._beta, self._thresholds)
