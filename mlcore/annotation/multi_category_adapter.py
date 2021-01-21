# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/annotation-multi_category_adapter.ipynb (unless otherwise specified).

__all__ = ['DEFAULT_ANNOTATIONS_FILE', 'logger', 'MultiCategoryAdapter', 'configure_logging']

# Cell

import csv
import sys
import argparse
import shutil
import logging
from os.path import join, basename, isfile, dirname
from ..io.core import create_folder
from .core import Annotation, AnnotationAdapter, Region

# Cell

DEFAULT_ANNOTATIONS_FILE = 'annotations.csv'

# Cell

logger = logging.getLogger(__name__)

# Cell


class MultiCategoryAdapter(AnnotationAdapter):
    """
    Adapter to read and write annotations for multi label classification.
    `args`: the arguments containing the parameters
    """

    def __init__(self, args):
        super().__init__(args)
        self.files_path = args.files_path
        self.annotations_file = args.annotations_file

    def read(self):
        """
        Read annotations from a multi classification CSV annotations file.
        return: the annotations as dictionary
        """
        annotations = {}

        logger.info('Read annotations from {}'.format(self.annotations_file))

        with open(self.annotations_file, newline='') as csv_file:
            reader = csv.DictReader(csv_file)

            skipped_annotations = []
            for row in reader:
                file_path = join(self.files_path, row['image_name'])
                if not isfile(file_path):
                    logger.warning("{}: Source file not found, skip annotation.".format(file_path))
                    skipped_annotations.append(file_path)
                    continue

                if file_path not in annotations:
                    annotations[file_path] = Annotation(annotation_id=file_path, file_path=file_path)

                annotation = annotations[file_path]

                tags = row['tags'] if 'tags' in row else []
                for category in tags.split(' '):
                    region = Region(labels=[category])
                    annotation.regions.append(region)

        logger.info('Finished read annotations')
        logger.info('Annotations read: {}'.format(len(annotations)))
        if skipped_annotations:
            logger.info('Annotations skipped: {}'.format(len(skipped_annotations)))
        return annotations

    def write(self, annotations):
        """
        Writes a multi classification CSV annotations file and copy the corresponding source files.
        `annotations`: the annotations to write
        """
        target_folder = create_folder(self.files_path)
        create_folder(dirname(self.annotations_file))
        logger.info('Write annotations to {}'.format(self.annotations_file))
        logger.info('Write file sources to {}'.format(target_folder))
        fieldnames = ['image_name', 'tags']
        with open(self.annotations_file, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=',', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            skipped_annotations = []
            for annotation in annotations.values():
                target_file = join(target_folder, basename(annotation.file_path))

                if not isfile(annotation.file_path):
                    logger.warning("{}: Source file not found, skip annotation.".format(annotation.file_path))
                    skipped_annotations.append(annotation.file_path)
                    continue
                if isfile(target_file):
                    logger.warning("{}: Target file already exist, skip annotation.".format(annotation.file_path))
                    skipped_annotations.append(annotation.file_path)
                    continue

                # copy the file
                shutil.copy2(annotation.file_path, target_file)
                writer.writerow({'image_name': basename(annotation.file_path),
                                 'tags': ' '.join(annotation.labels())})

        logger.info('Finished write annotations')
        logger.info('Annotations written: {}'.format(len(annotations) - len(skipped_annotations)))
        if skipped_annotations:
            logger.info('Annotations skipped: {}'.format(len(skipped_annotations)))

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
        parser.add_argument(cls.assign_prefix('--annotations_file', prefix),
                            dest="annotations_file",
                            help="The path to the multi classification CSV annotation file.",
                            required=True)
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
