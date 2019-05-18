import numpy as np


def calc_boxes_areas(boxes: np.ndarray):
    """
    Calculate areas of array of boxes

    :param boxes: array of boxes with shape [B, N, 4]
    :return: array of boxes area with shape [B, N]
    """
    xx, yy = np.take(boxes, [0, 2], axis=2), np.take(boxes, [1, 3], axis=2)  # [B, N, 2], [B, N, 2]
    boxes_x_min, boxes_x_max = xx.min(2), xx.max(2)  # [B, N], [B, N]
    boxes_y_min, boxes_y_max = yy.min(2), yy.max(2)  # [B, N], [B, N]
    return (boxes_x_max - boxes_x_min) * (boxes_y_max - boxes_y_min)  # [B, N], [B, N]
