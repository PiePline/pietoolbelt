import numpy as np


def calc_boxes_areas(boxes: np.ndarray):
    """
    Calculate areas of array of boxes

    :param boxes: array of boxes with shape [N, 4]
    :return: array of boxes area with shape [N]
    """
    xx, yy = np.take(boxes, [0, 2], axis=1), np.take(boxes, [1, 3], axis=1)  # [N, 2], [N, 2]
    boxes_x_min, boxes_x_max = xx.min(1), xx.max(1)  # [N], [N]
    boxes_y_min, boxes_y_max = yy.min(1), yy.max(1)  # [N], [N]
    return (boxes_x_max - boxes_x_min) * (boxes_y_max - boxes_y_min)  # [N], [N]
