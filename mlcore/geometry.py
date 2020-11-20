# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/geometry.ipynb (unless otherwise specified).

__all__ = ['create_ellipse', 'ellipse_intersection_area']

# Cell

from shapely.geometry.point import Point
from shapely import affinity

# Cell


def create_ellipse(center, lengths, angle=0):
    """
    Create a shapely ellipse.
    Adapted from https://gis.stackexchange.com/a/243462
    `center`: a tuple with the center x and y coordinates
    `lengths`: a tuple with the x and y lengths
    `angle`: a rotation angle
    return: the rotated ellipse
    """
    circ = Point(center).buffer(1)
    ell = affinity.scale(circ, int(lengths[0]), int(lengths[1]))
    ellr = affinity.rotate(ell, angle)
    return ellr

# Cell


def ellipse_intersection_area(ellipse1, ellipse2):
    """
    Calculates the intersection of two ellipses.
    Adapted from https://stackoverflow.com/a/48812832
    `ellipse1`: the first shapely ellipse
    `ellipse2`: the second shapely ellipse
    return: the intersection area
    """
    intersect = ellipse1.intersection(ellipse2)
    return intersect.area