# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/evaluation-core.ipynb (unless otherwise specified).

__all__ = ['logger', 'box_area', 'intersection_box', 'union_box', 'intersection_over_union', 'configure_logging']

# Cell

import argparse
import logging
import sys

# Cell

logger = logging.getLogger(__name__)

# Cell


def box_area(box):
    """
    Calculates the area of a bounding box.
    Source code mainly taken from:
    https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    `box`: the bounding box to calculate the area for with the format ((x_min, x_max), (y_min, y_max))
    return: the bounding box area
    """
    return max(0, box[0][1] - box[0][0] + 1) * max(0, box[1][1] - box[1][0] + 1)

# Cell


def intersection_box(box_a, box_b):
    """
    Calculates the intersection box from two bounding boxes with the format ((x_min, x_max), (y_min, y_max)).
    Source code mainly taken from:
    https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    `box_a`: the first box
    `box_b`: the second box
    return: the intersection box
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = max(box_a[0][0], box_b[0][0])
    y_a = max(box_a[1][0], box_b[1][0])
    x_b = min(box_a[0][1], box_b[0][1])
    y_b = min(box_a[1][1], box_b[1][1])
    return (x_a, x_b), (y_a, y_b)

# Cell


def union_box(box_a, box_b):
    """
    Calculates the union box from two bounding boxes with the format ((x_min, x_max), (y_min, y_max)).
    Source code mainly taken from:
    https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    `box_a`: the first box
    `box_b`: the second box
    return: the union box
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    x_a = min(box_a[0][0], box_b[0][0])
    y_a = min(box_a[1][0], box_b[1][0])
    x_b = max(box_a[0][1], box_b[0][1])
    y_b = max(box_a[1][1], box_b[1][1])
    return (x_a, x_b), (y_a, y_b)

# Cell


def intersection_over_union(box_a, box_b):
    """
    Intersection over Union (IoU) algorithm.
    Calculates the IoU from two bounding boxes with the format ((x_min, x_max), (y_min, y_max)).
    Source code mainly taken from:
    https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    `box_a`: the first box
    `box_b`: the second box
    return: the IoU
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    inter_box = intersection_box(box_a, box_b)
    # compute the area of intersection rectangle
    inter_area = box_area(inter_box)
    # compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = box_area(box_a)
    box_b_area = box_area(box_b)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    # return the intersection over union value
    return iou

# Cell


def configure_logging(logging_level=logging.INFO):
    """
    Configures logging for the system.

    :param logging_level: The logging level to use.
    """
    logger.setLevel(logging_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging_level)

    logger.addHandler(handler)

# Cell


if __name__ == '__main__' and '__file__' in globals():
    # for direct shell execution
    configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("annotation",
                        help="The path to the VIA annotation file.")

    args = parser.parse_args()
