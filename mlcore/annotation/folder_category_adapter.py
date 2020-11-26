# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/annotation-folder_category_adapter.ipynb (unless otherwise specified).

__all__ = ['DEFAULT_CATEGORY_FOLDER_INDEX', 'logger', 'read_annotations', 'write_annotations', 'configure_logging']

# Cell

import sys
import argparse
import logging
import shutil
from os.path import join, normpath, sep, getsize, basename
from ..io.core import create_folder, scan_files
from .core import Annotation, Region, create_annotation_id

# Cell

DEFAULT_CATEGORY_FOLDER_INDEX = -2

# Cell

logger = logging.getLogger(__name__)

# Cell


def read_annotations(files_source, category_index=DEFAULT_CATEGORY_FOLDER_INDEX):
    """
    Read annotations from folder structure representing the categories.
    `files_source`: the path to the folder containing subfolders as category label with source files
    return: the annotations
    """
    annotations = {}
    file_paths = scan_files(files_source)

    for file_path in file_paths:
        annotation_id = create_annotation_id(file_path)
        if annotation_id not in annotations:
            file_size = getsize(file_path)
            file_name = basename(file_path)
            annotations[annotation_id] = Annotation(annotation_id=annotation_id, file_name=file_name,
                                                    file_size=file_size, file_path=file_path)
        annotation = annotations[annotation_id]

        trimmed_path = _trim_base_path(file_path, files_source)
        path_split = normpath(trimmed_path).lstrip(sep).split(sep)

        if len(path_split) <= abs(category_index):
            logger.warning("{}: No category folder found, skip annotations.".format(file_path))
            continue

        category = normpath(trimmed_path).lstrip(sep).split(sep)[category_index]
        region = Region(labels=[category])
        annotation.regions.append(region)

    return annotations

# Cell


def write_annotations(target_path, annotations):
    """
    Write annotations to folder structure representing the categories.
    The category folder will is created, if not exist, and corresponding files are copied into the labeled folder.
    `target_path`: the target path to create labeled folder structure into
    `annotations`: the annotations to write
    """
    for annotation in annotations.values():
        for label in annotation.labels():
            category_folder = create_folder(join(target_path, label))
            shutil.copy2(annotation.file_path, join(category_folder, annotation.file_name))

# Cell


def _trim_base_path(file_path, base_path):
    """
    Trims the base path from a file path.
    `file_path`: the file path to trim from
    `base_path`: the base path to trim
    return: the trimmed file path
    """
    if file_path.startswith(base_path):
        file_path = file_path[len(base_path):]
    return file_path

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
    parser.add_argument("files-source",
                        help="The path to the folder containing the source files.")
    parser.add_argument("--category-index",
                        help="The folder index, representing the category.",
                        type=int,
                        default=DEFAULT_CATEGORY_FOLDER_INDEX)

    args = parser.parse_args()

    read_annotations(args.files_source, args.category_index)
