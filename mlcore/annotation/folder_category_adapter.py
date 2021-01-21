# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/annotation-folder_category_adapter.ipynb (unless otherwise specified).

__all__ = ['DEFAULT_CATEGORY_FOLDER_INDEX', 'logger', 'FolderCategoryAdapter', 'configure_logging']

# Cell

import sys
import argparse
import logging
import shutil
from os.path import join, normpath, sep, basename, isfile
from ..io.core import create_folder, scan_files
from .core import Annotation, AnnotationAdapter, Region

# Cell

DEFAULT_CATEGORY_FOLDER_INDEX = -2

# Cell

logger = logging.getLogger(__name__)

# Cell


class FolderCategoryAdapter(AnnotationAdapter):
    """
    Adapter to read and write annotations where the folder structure represents the categories.
    `args`: the arguments containing the parameters
    """

    def __init__(self, args):
        super().__init__()
        self.files_path = args.files_path
        self.category_index = args.category_index if args.category_index is not None else DEFAULT_CATEGORY_FOLDER_INDEX

    def read(self):
        """
        Read annotations from folder structure representing the categories.
        return: the annotations as dictionary
        """
        annotations = {}

        logger.info('Read annotations from {}'.format(self.files_path))

        file_paths = scan_files(self.files_path)

        skipped_annotations = []
        for file_path in file_paths:
            trimmed_path = self._trim_base_path(file_path, self.files_path)
            if trimmed_path not in annotations:
                annotations[trimmed_path] = Annotation(annotation_id=trimmed_path, file_path=file_path)
            annotation = annotations[trimmed_path]

            path_split = normpath(trimmed_path).lstrip(sep).split(sep)

            if len(path_split) <= abs(self.category_index):
                logger.warning("{}: No category folder found, skip annotation.".format(trimmed_path))
                skipped_annotations.append(file_path)
                continue

            category = path_split[self.category_index]
            region = Region(labels=[category])
            annotation.regions.append(region)

        logger.info('Finished read annotations')
        logger.info('Annotations read: {}'.format(len(annotations)))
        if skipped_annotations:
            logger.info('Annotations skipped: {}'.format(len(skipped_annotations)))
        return annotations

    def write(self, annotations):
        """
        Write annotations to folder structure representing the categories.
        The category folder is created, if not exist, and corresponding files are copied into the labeled folder.
        `annotations`: the annotations to write
        """
        logger.info('Write annotations to {}'.format(self.files_path))
        skipped_annotations = []
        for annotation in annotations.values():
            if not isfile(annotation.file_path):
                logger.warning("{}: Source file not found, skip annotation.".format(annotation.file_path))
                skipped_annotations.append(annotation.file_path)
                continue

            skipped_labels = []
            for label in annotation.labels():
                category_folder = create_folder(join(self.files_path, label))
                target_file = join(category_folder, basename(annotation.file_path))
                if isfile(target_file):
                    logger.warning("{}: Target file already exist, skip label {}.".format(annotation.file_path, label))
                    skipped_labels.append(label)
                    continue
                shutil.copy2(annotation.file_path, target_file)
            if len(skipped_labels) == len(annotation.labels):
                logger.warning("{}: All labels skipped, skip annotation.".format(annotation.file_path))
                skipped_annotations.append(annotation.file_path)

        logger.info('Finished write annotations')
        logger.info('Annotations written: {}'.format(len(annotations) - len(skipped_annotations)))
        if skipped_annotations:
            logger.info('Annotations skipped: {}'.format(len(skipped_annotations)))

    @classmethod
    def _trim_base_path(cls, file_path, base_path):
        """
        Trims the base path from a file path.
        `file_path`: the file path to trim from
        `base_path`: the base path to trim
        return: the trimmed file path
        """
        if file_path.startswith(base_path):
            file_path = file_path[len(base_path):]
        return file_path

    @classmethod
    def argparse(cls, prefix=None):
        """
        Returns the argument parser containing argument definition for command line use.
        `prefix`: a parameter prefix to set, if needed
        return: the argument parser
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(cls.assign_prefix('--files_path', prefix),
                            dest="files_path",
                            help="The path to the folder containing the files.",
                            required=True)
        parser.add_argument(cls.assign_prefix('--category_index', prefix),
                            dest="category_index",
                            help="The folder index, representing the category.",
                            type=int,
                            default=DEFAULT_CATEGORY_FOLDER_INDEX)
        return parser

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
