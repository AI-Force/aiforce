# AUTOGENERATED! DO NOT EDIT! File to edit: annotation-multi_category_adapter.ipynb (unless otherwise specified).

__all__ = ['DEFAULT_ANNOTATIONS_FILE', 'CSV_FIELDNAMES', 'logger', 'MultiCategoryAnnotationAdapter']


# Cell
import csv
import shutil
import logging
from os.path import join, basename, isfile, splitext
from ..core import assign_arg_prefix
from ..io.core import create_folder
from .core import Annotation, AnnotationAdapter, Region, SubsetType


# Cell
DEFAULT_ANNOTATIONS_FILE = 'annotations.csv'
CSV_FIELDNAMES = ['image_name', 'tags']


# Cell
logger = logging.getLogger(__name__)


# Cell
class MultiCategoryAnnotationAdapter(AnnotationAdapter):
    """
    Adapter to read and write annotations for multi label classification.
    """

    def __init__(self, path, categories_file_name=None, annotations_file_name=None):
        """
        Multi Label Classification Adapter to read and write annotations.
        `path`: the folder containing the annotations
        `categories_file_name`: the name of the categories file
        `annotations_file_name`: the name of annotations file
        """
        super().__init__(path, categories_file_name)

        if annotations_file_name is None:
            self.annotations_file_name = DEFAULT_ANNOTATIONS_FILE
        else:
            self.annotations_file_name = annotations_file_name

    @classmethod
    def argparse(cls, prefix=None):
        """
        Returns the argument parser containing argument definition for command line use.
        `prefix`: a parameter prefix to set, if needed
        return: the argument parser
        """
        parser = super(MultiCategoryAnnotationAdapter, cls).argparse(prefix=prefix)
        parser.add_argument(assign_arg_prefix('--annotations_file', prefix),
                            dest="annotations_file_name",
                            help="The name of the multi classification CSV annotation file.",
                            default=None)
        return parser

    def read_annotations(self, categories, subset_type=SubsetType.TRAINVAL):
        """
        Read annotations from a multi classification CSV annotations file.
        `categories`: the categories as list
        `subset_type`: the subset type to read
        return: the annotations as dictionary
        """
        path = join(self.path, str(subset_type))
        annotations_file_name = self._annotation_file_name_suffix_handling(subset_type)
        annotations_file_path = join(self.path, annotations_file_name)
        logger.info('Read file sources from {}'.format(path))
        logger.info('Read annotations from {}'.format(annotations_file_path))

        annotations = {}

        with open(annotations_file_path, newline='') as csv_file:
            reader = csv.DictReader(csv_file)

            skipped_annotations = []
            for row in reader:
                file_path = join(path, row['image_name'])
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

    def write_annotations(self, annotations, categories, subset_type=SubsetType.TRAINVAL):
        """
        Writes a multi classification CSV annotations file and copy the corresponding source files.
        `annotations`: the annotations as dictionary
        `categories`: the categories as list
        `subset_type`: the subset type to write
        return: a list of written target file paths
        """
        path = join(self.path, str(subset_type))
        target_folder = create_folder(path)
        annotations_file_name = self._annotation_file_name_suffix_handling(subset_type)
        annotations_file_path = join(self.path, annotations_file_name)
        logger.info('Write file sources to {}'.format(target_folder))
        logger.info('Write annotations to {}'.format(annotations_file_path))

        skipped_annotations = []
        copied_files = []
        rows = {}
        for annotation in annotations.values():
            target_file_name = basename(annotation.file_path)
            target_file = join(target_folder, target_file_name)

            if not isfile(annotation.file_path):
                logger.warning("{}: Source file not found, skip annotation.".format(annotation.file_path))
                skipped_annotations.append(annotation.file_path)
                continue

            # copy the file
            if target_file_name not in rows:
                rows[target_file_name] = []
            rows[target_file_name].extend(annotation.labels())

            if isfile(target_file):
                logger.warning("{}: Target file already exist, skip annotation.".format(annotation.file_path))
                skipped_annotations.append(annotation.file_path)
                continue
            shutil.copy2(annotation.file_path, target_file)
            copied_files.append(target_file)
        with open(annotations_file_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES, delimiter=',', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            writer.writerows([dict(zip(CSV_FIELDNAMES, [key, ' '.join(labels)])) for key, labels in rows.items()])

        logger.info('Finished write annotations')
        logger.info('Annotations written: {}'.format(len(annotations) - len(skipped_annotations)))
        if skipped_annotations:
            logger.info('Annotations skipped: {}'.format(len(skipped_annotations)))
        return copied_files

    def _annotation_file_name_suffix_handling(self, subset_type):
        """
        Handle annotations file name based on the subset type.
        `subset_type`: the subset type to handle
        return: the annotations file name
        """
        file_name, ext = splitext(self.annotations_file_name)
        if subset_type in [SubsetType.TRAIN, SubsetType.VAL] and not file_name.endswith(str(subset_type)):
            suffix = "_{}".format(str(subset_type))
            return "{}{}{}".format(file_name, suffix, ext)
        return self.annotations_file_name
