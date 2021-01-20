# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/annotation-core.ipynb (unless otherwise specified).

__all__ = ['RegionShape', 'parse_region_shape', 'Region', 'Annotation', 'create_annotation_id', 'convert_region',
           'region_bounding_box', 'AnnotationAdapter']

# Cell

import re
from abc import ABC, abstractmethod
from enum import Enum
from copy import deepcopy
from os.path import isfile
from ..io.core import get_file_sha

# Cell


class RegionShape(Enum):
    """
    The supported region shape types
    """
    NONE = 'none'
    CIRCLE = 'circle'
    ELLIPSE = 'ellipse'
    POINT = 'point'
    POLYGON = 'polygon'
    RECTANGLE = 'rect'

    def __str__(self):
        return self.value

# Cell


def parse_region_shape(shape_str):
    """
    Try to parse the region shape from a string representation.
    `shape_str`: the shape as string
    return: the parsed RegionShape
    raises: `ValueError` if unsupported shape parsed
    """
    try:
        return RegionShape(shape_str)
    except ValueError:
        raise ValueError("Error, unsupported region shape: {}".format(shape_str))

# Cell


class Region:
    """
    A region
    `shape`: the region shape
    `points_x`: a list of points x-coordinates
    `points_y`: a list of points y-coordinates
    `radius_x`: a radius on x-coordinate
    `radius_y`: a radius on y-coordinate
    `labels`: a set of region labels
    """

    def __init__(self, shape=RegionShape.NONE, points_x=None, points_y=None, radius_x=0, radius_y=0, labels=None):
        self.shape = shape
        self.points_x = [] if points_x is None else points_x
        self.points_y = [] if points_y is None else points_y
        self.radius_x = radius_x
        self.radius_y = radius_y
        self.labels = [] if labels is None else labels

# Cell


class Annotation:
    """
    A annotation for a file.
    `annotation_id`: a unique annotation identifier
    `file_name`: the file
    `file_path`: the file path
    `regions`: A list of regions
    """

    def __init__(self, annotation_id=None, file_path=None, regions=None):
        self.annotation_id = annotation_id
        self.file_path = file_path
        self.regions: [Region] = [] if regions is None else regions

    def labels(self):
        """
        Returns a list of labels, assigned to the annotation.
        return: a list of labels
        """
        labels = {}
        for region in self.regions:
            for label in region.labels:
                labels[label] = None
        return list(labels.keys())

# Cell


def create_annotation_id(file_path):
    """
    Creates a annotation ID
    `file_path`: the file_path to create the ID from
    return: the ID if file exist, else None
    """
    if not isfile(file_path):
        return None
    sha1 = get_file_sha(file_path)
    return sha1

# Cell


def convert_region(region: Region, target_shape: RegionShape):
    """
    Convert region to target shape.
    `region`: the region to convert
    `target_shape`: the target shape to convert to
    """
    if target_shape != region.shape:
        x_min = min(region.points_x) - region.radius_x if len(region.points_x) else 0
        x_max = max(region.points_x) + region.radius_x if len(region.points_x) else 0
        y_min = min(region.points_y) - region.radius_y if len(region.points_y) else 0
        y_max = max(region.points_y) + region.radius_y if len(region.points_y) else 0
        center_x = x_min + x_max - x_min
        center_y = y_min + y_max - y_min
        region.shape = target_shape
        if target_shape == RegionShape.NONE:
            region.points_x = []
            region.points_y = []
            region.radius_x = 0
            region.radius_y = 0
        elif target_shape == RegionShape.CIRCLE or target_shape == RegionShape.ELLIPSE:
            region.points_x = [center_x]
            region.points_y = [center_y]
            region.radius_x = x_max - center_x
            region.radius_y = y_max - center_y
        elif target_shape == RegionShape.POINT:
            region.points_x = [center_x]
            region.points_y = [center_y]
            region.radius_x = 0
            region.radius_y = 0
        elif target_shape == RegionShape.POLYGON:
            region.points_x = [x_min, x_min, x_max, x_max, x_min]
            region.points_y = [y_min, y_max, y_max, y_min, y_min]
            region.radius_x = 0
            region.radius_y = 0
        elif target_shape == RegionShape.RECTANGLE:
            region.points_x = [x_min, x_max]
            region.points_y = [y_min, y_max]
            region.radius_x = 0
            region.radius_y = 0
        else:
            raise NotImplementedError('unsupported conversion {} -> {}'.format(region.shape, target_shape))

# Cell


def region_bounding_box(region: Region):
    """
    Calculates the region bounding box.
    `region`: the region
    return: a tuple of points_x and points_y
    """
    bbox = deepcopy(region)
    convert_region(bbox, RegionShape.RECTANGLE)
    return bbox.points_x, bbox.points_y

# Cell


class AnnotationAdapter(ABC):
    """
    Abstract Base Adapter to inherit for writing custom adapters
    """

    @abstractmethod
    def read(self):
        """
        Read annotations.
        return: the annotations as dictionary
        """
        pass

    @abstractmethod
    def write(self, annotations):
        """
        Write annotations.
        `annotations`: the annotations to write
        """
        pass

    @classmethod
    @abstractmethod
    def argparse(cls, prefix=None):
        """
        Returns the argument parser containing argument definition for command line use.
        `prefix`: a parameter prefix to set, if needed
        return: the argument parser
        """
        pass

    @classmethod
    def assign_prefix(cls, arg_name, prefix=None):
        """
        Assign a parameter prefix to a given argument name. (e.g --prefix_<arg_name>)
        `arg_name`: the argument name to prefix
        `prefix`: the prefix
        return: the prefixed argument name
        """
        return re.sub(r'^(-{0,2})([\w-]+)$', r'\1{}_\2'.format('' if prefix is None else prefix), arg_name)
