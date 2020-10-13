# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/annotation-via.ipynb (unless otherwise specified).

__all__ = ['DEFAULT_ANNOTATIONS_FILE', 'DEFAULT_CATEGORY_LABEL_KEY', 'logger', 'Shape', 'read_annotations',
           'create_annotation_id', 'get_annotation_for_file', 'get_regions', 'set_regions', 'get_region_attributes',
           'get_region_category', 'get_shape_attributes', 'get_shape_type', 'get_points', 'set_points',
           'convert_region_rect_to_polygon', 'convert_region_polygon_to_rect', 'configure_logging']

# Cell

import json
import sys
import argparse
import logging
from enum import Enum
from os.path import basename, isfile, getsize

# Cell

DEFAULT_ANNOTATIONS_FILE = 'via_region_data.json'
DEFAULT_CATEGORY_LABEL_KEY = 'category'

# Cell

logger = logging.getLogger(__name__)

# Cell


class Shape(Enum):
    """
    The supported VIA shape types
    """
    RECTANGLE = 'rect'
    POLYGON = 'polygon'

    def __str__(self):
        return self.value

# Cell


def read_annotations(annotations_file):
    """
    Reads an VIA annotation file
    `annotations_file`: the path to the annotation file to read
    return: the annotations
    """
    with open(annotations_file) as json_file:
        annotations = json.load(json_file)
        logger.info('Found {} annotations at {}'.format(len(annotations), annotations_file))

    return annotations

# Cell


def create_annotation_id(file_path):
    """
    Creates a VIA annotation ID
    `file_path`: the file_path to create the ID from
    return: the ID if file exist, else None
    """
    if not isfile(file_path):
        return None

    filename = basename(file_path)
    file_size = getsize(file_path)
    return '{:s}{:d}'.format(filename, file_size)

# Cell


def get_annotation_for_file(annotations, file_path):
    """
    Finds VIA annotation to a file_path
    `annotations`: the annotations to search in
    `file_path`: the file_path to search for
    return: the annotation if found, else None
    """
    annotation_id = create_annotation_id(file_path)
    return annotations[annotation_id] if annotation_id is not None and annotation_id in annotations else None

# Cell


def get_regions(annotation):
    """
    Get the regions information from an annotation.
    `annotation`: annotation to get the regions from
    return: a dictionary of the annotation regions
    """
    return annotation['regions'] if annotation and 'regions' in annotation else {}

# Cell


def set_regions(annotation, regions):
    """
    Set the regions information to an annotation.
    `annotation`: annotation to set the regions
    `regions`: the regions to set
    """
    if annotation:
        annotation['regions'] = regions

# Cell


def get_region_attributes(region):
    """
    Get the region attributes information from a region.
    `region`: region to get the shape attributes for
    return: a dictionary of the region attributes or None if no region attributes exist
    """
    return region['region_attributes'] if region and 'region_attributes' in region else None

# Cell


def get_region_category(region_attributes, key=DEFAULT_CATEGORY_LABEL_KEY):
    """
    Get the region category from a region attributes.
    `region_attributes`: region_attributes to get the category from
    `key`: if a custom category label key exist
    return: the region category if exist, else none
    """
    return region_attributes[key] if region_attributes and key in region_attributes else None

# Cell


def get_shape_attributes(region):
    """
    Get the shape attributes information from a region.
    `region`: region to get the shape attributes for
    return: a dictionary of the shape attributes or None if no shape attributes exist
    """
    return region['shape_attributes'] if region and 'shape_attributes' in region else None

# Cell


def get_shape_type(shape_attributes):
    """
    Get the shape type from shape attributes.
    `shape_attributes`: shape_attributes to get the shape type for
    return: The supported shape type if exist, else None
    """
    shape_type_str = shape_attributes['name'] if shape_attributes and 'name' in shape_attributes else None
    try:
        shape_type = Shape(shape_type_str)
    except ValueError:
        return None

    return shape_type

# Cell


def get_points(shape_attributes):
    """
    Get the points of a shape.
    `shape_attributes`: shape_attributes to get the points for
    return: a tuple of arrays of x-points and y-points
    """
    shape_type = get_shape_type(shape_attributes)
    x_points = []
    y_points = []
    if shape_type == Shape.RECTANGLE:
        x = shape_attributes["x"]
        y = shape_attributes["y"]
        max_x = x + shape_attributes["width"]
        max_y = y + shape_attributes["height"]
        x_points = [x, max_x]
        y_points = [y, max_y]
    elif shape_type == Shape.POLYGON:
        x_points = shape_attributes['all_points_x']
        y_points = shape_attributes['all_points_y']
    return x_points, y_points

# Cell


def set_points(shape_attributes, x_points, y_points):
    """
    Set the points of a shape.
    `shape_attributes`: shape_attributes to set the points
    `x-points`: the x_points to set
    `y-points`: the y_points to set
    """
    shape_type = get_shape_type(shape_attributes)
    if shape_type == Shape.RECTANGLE:
        shape_attributes["x"] = x_points[0]
        shape_attributes["y"] = y_points[0]
        shape_attributes["width"] = x_points[1] - x_points[0]
        shape_attributes["height"] = y_points[1] - y_points[0]
    elif shape_type == Shape.POLYGON:
        shape_attributes['all_points_x'] = x_points
        shape_attributes['all_points_y'] = y_points

# Cell


def convert_region_rect_to_polygon(region):
    """
    Converts a region from rectangle to polygon.
    `region`: region to convert
    return: the converted region
    """
    shape_attributes = region['shape_attributes']
    shape_type = get_shape_type(shape_attributes)
    if shape_type == Shape.RECTANGLE:
        x = shape_attributes["x"]
        y = shape_attributes["y"]
        max_x = x + shape_attributes["width"]
        max_y = y + shape_attributes["height"]
        shape_attributes = {
            "name": Shape.POLYGON.value,
            "all_points_x": [x, x, max_x, max_x, x],
            "all_points_y": [y, max_y, max_y, y, y],
        }
        region['shape_attributes'] = shape_attributes
    return region

# Cell


def convert_region_polygon_to_rect(region):
    """
    Converts a region from polygon to rectangle.
    `region`: region to convert
    return: the converted region
    """
    shape_attributes = get_shape_attributes(region)
    shape_type = get_shape_type(shape_attributes)
    if shape_type == Shape.POLYGON:
        x_min = min(shape_attributes['all_points_x'])
        x_max = max(shape_attributes['all_points_x'])
        y_min = min(shape_attributes['all_points_y'])
        y_max = max(shape_attributes['all_points_y'])
        shape_attributes = {
            "name": Shape.RECTANGLE.value,
            "x": x_min,
            "y": y_min,
            "width": x_max - x_min,
            "height": y_max - y_min,
        }
        region['shape_attributes'] = shape_attributes
    return region

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

    read_annotations(args.annotation)
