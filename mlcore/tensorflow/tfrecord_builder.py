# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/tensorflow-tfrecord_builder.ipynb (unless otherwise specified).

__all__ = ['logger', 'create_tfrecord_entry', 'create_tfrecord_file', 'int64_feature', 'int64_list_feature',
           'bytes_feature', 'bytes_list_feature', 'float_list_feature', 'create_labelmap_file', 'configure_logging']

# Cell

import logging
import logging.handlers
import argparse
import io
import sys
import tensorflow as tf
from os import environ
from os.path import join
from ..core import Type, infer_type
from ..annotation.via import read_annotations
from ..image.pillow_tools import get_image_size
from mlcore import category_tools

# Cell

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)

logger = logging.getLogger(__name__)

# Cell


def create_tfrecord_entry(categories, file_annotation):
    """
    Create tfrecord entry with annotations for one file / image.
    `categories`: the categories used
    `file_annotation`: the annotation of a file / image
    return: the tfrecord entry
    """
    with tf.io.gfile.GFile(file_annotation.file_path, 'rb') as fid:
        encoded_jpg = fid.read()
    _, width, height = get_image_size(io.BytesIO(encoded_jpg))

    file_name = file_annotation.file_name.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, annotation in enumerate(file_annotation.annotations):
        x_min = min(annotation.points_x)
        y_min = min(annotation.points_y)
        x_max = max(annotation.points_x)
        y_max = max(annotation.points_y)
        category = annotation.labels[0] if len(annotation.labels) else ''

        xmins.append(x_min / width)
        xmaxs.append(x_max / width)
        ymins.append(y_min / height)
        ymaxs.append(y_max / height)
        classes_text.append(category.encode('utf8'))
        classes.append(categories.index(category))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(file_name),
        'image/source_id': bytes_feature(file_name),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))
    return tf_example

# Cell


def create_tfrecord_file(output_path, categories, annotations):
    """
    Create a tfrecord file for a sub-data-set, which can be one of the following: training, validation, test
    `output_path`: the path including the filename of the tfrecord file
    `categories`: the categories used
    `annotations`: the annotations of the files / images
    """
    writer = tf.io.TFRecordWriter(output_path)
    for annotation in annotations.values():
        tf_example = create_tfrecord_entry(categories, annotation)
        writer.write(tf_example.SerializeToString())
    writer.close()
    logger.info('Successfully created the TFRecord file: {}'.format(output_path))

# Cell


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# Cell


def create_labelmap_file(output_path, categories, start=1):
    """
    Create labelmap protobuffer text file containing the categories.
    Format is compatible with Tensorflow Object Detection API.
    For object detection data-sets, the categories should exclude the background class and `start` should be 1.
    `output_path`: the path including the filename of the protobuffer text file
    `categories`: a list of the categories to write
    `start`: the category index for the first category
    """
    # create label_map data
    label_map = ''
    for index, category in enumerate(categories, start=start):
        label_map = label_map + "item {\n"
        label_map = label_map + " id: " + str(index) + "\n"
        label_map = label_map + " name: '" + category + "'\n}\n\n"
    label_map = label_map[:-1]

    # write label_map file
    with open(output_path, "w") as f:
        f.write(label_map)
        f.close()

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
    configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("-o",
                        "--output",
                        help="Path of output TFRecord (.record) file.",
                        type=str)
    parser.add_argument("-s",
                        "--source",
                        help="The path to the data-set source files.",
                        type=str)
    parser.add_argument("-c",
                        "--categories",
                        help="The path to the data-set categories file.",
                        type=str)
    parser.add_argument("-a",
                        "--annotation",
                        help="The path to the data-set annotation file, the data-set is build from.",
                        type=str)
    parser.add_argument("-t",
                        "--type",
                        help="The type of the data-set, if not explicitly set try to infer from categories file path.",
                        choices=list(Type),
                        type=Type,
                        default=None)
    args = parser.parse_args()

    category_file_path = args.categories
    data_set_type = args.type
    # try to infer the data-set type if not explicitly set
    if data_set_type is None:
        try:
            data_set_type = infer_type(category_file_path)
        except ValueError as e:
            logger.error(e)
            sys.exit(1)

    categories = category_tools.read_categories(category_file_path, data_set_type)
    annotations = read_annotations(args.annotation, args.source)

    create_tfrecord_file(args.output, categories, annotations)

    logger.info('FINISHED!!!')
