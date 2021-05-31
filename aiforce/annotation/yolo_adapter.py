# AUTOGENERATED! DO NOT EDIT! File to edit: annotation-yolo_adapter.ipynb (unless otherwise specified).

__all__ = ['FIELD_NAMES', 'DEFAULT_IMAGES_FOLDER', 'DEFAULT_IMAGE_ANNOTATIONS_FOLDER', 'logger',
           'YOLOAnnotationAdapter']


# Cell
import csv
import logging
import shutil
from os.path import join, splitext, basename, isfile
from mlcore.category_tools import read_categories
from mlcore.core import assign_arg_prefix
from mlcore.io.core import scan_files, create_folder
from mlcore.image.pillow_tools import get_image_size
from mlcore.annotation.core import Annotation, AnnotationAdapter, Region, RegionShape, SubsetType


# Cell
FIELD_NAMES = ['class_number', 'c_x', 'c_y', 'width', 'height']
DEFAULT_IMAGES_FOLDER = 'images'
DEFAULT_IMAGE_ANNOTATIONS_FOLDER = 'labels'


# Cell
logger = logging.getLogger(__name__)


# Cell
class YOLOAnnotationAdapter(AnnotationAdapter):
    """
    Adapter to read and write annotations in the YOLO format.
    """

    def __init__(self, path, categories_file_name=None, images_folder_name=None, annotations_folder_name=None):
        """
        YOLO Adapter to read and write annotations.
        `path`: the folder containing the annotations
        `categories_file_name`: the name of the categories file
        `images_folder_name`: the name of the folder containing the image files
        `annotations_folder_name`: the name of the folder containing the mage annotations
        """
        super().__init__(path, categories_file_name)

        if images_folder_name is None:
            self.images_folder_name = DEFAULT_IMAGES_FOLDER
        else:
            self.images_folder_name = images_folder_name

        if annotations_folder_name is None:
            self.annotations_folder_name = DEFAULT_IMAGE_ANNOTATIONS_FOLDER
        else:
            self.annotations_folder_name = annotations_folder_name

    @classmethod
    def argparse(cls, prefix=None):
        """
        Returns the argument parser containing argument definition for command line use.
        `prefix`: a parameter prefix to set, if needed
        return: the argument parser
        """
        parser = super(YOLOAnnotationAdapter, cls).argparse(prefix=prefix)
        parser.add_argument(assign_arg_prefix('--images_folder_name', prefix),
                            dest="images_folder_name",
                            help="The name of the folder containing the image files.",
                            default=None)
        parser.add_argument(assign_arg_prefix('--annotations_folder_name', prefix),
                            dest="annotations_folder_name",
                            help="The name of the folder containing the mage annotations.",
                            default=None)
        return parser

    def read_annotations(self, subset_type=SubsetType.TRAINVAL):
        """
        Reads YOLO annotations.
        `subset_type`: the subset type to read
        return: the annotations as dictionary
        """
        path = join(self.path, str(subset_type))
        annotations = {}
        annotations_path = join(path, self.annotations_folder_name)
        images_path = join(path, self.images_folder_name)
        logger.info('Read images from {}'.format(images_path))
        logger.info('Read annotations from {}'.format(annotations_path))

        annotation_files = scan_files(annotations_path, file_extensions='.txt')
        categories = read_categories(join(self.path, self.categories_file_name))
        categories_len = len(categories)
        skipped_annotations = []

        for annotation_file in annotation_files:
            with open(annotation_file, newline='') as csv_file:
                annotation_file_name = basename(annotation_file)
                file_name, _ = splitext(annotation_file_name)
                file_path = join(images_path, '{}{}'.format(file_name, '.jpg'))

                if not isfile(file_path):
                    logger.warning("{}: Source file not found, skip annotation.".format(file_path))
                    skipped_annotations.append(file_path)
                    continue

                if annotation_file not in annotations:
                    annotations[annotation_file] = Annotation(annotation_id=annotation_file, file_path=file_path)

                annotation = annotations[annotation_file]

                reader = csv.DictReader(csv_file, fieldnames=FIELD_NAMES, delimiter=' ')
                _, image_width, image_height = get_image_size(file_path)
                for row in reader:
                    c_x = float(row["c_x"])
                    c_y = float(row["c_y"])
                    width = float(row["width"])
                    height = float(row["height"])
                    class_number = int(row["class_number"])
                    # denormalize bounding box
                    x_min = self._denormalize_value(c_x - (width / 2), image_width)
                    y_min = self._denormalize_value(c_y - (height / 2), image_height)
                    x_max = x_min + self._denormalize_value(width, image_width)
                    y_max = y_min + self._denormalize_value(height, image_height)
                    points_x = [x_min, x_max]
                    points_y = [y_min, y_max]

                    labels = [categories[class_number]] if class_number < categories_len else []
                    if not labels:
                        logger.warning("{}: Class number exceeds categories, set label as empty.".format(
                            annotation_file
                        ))
                    region = Region(shape=RegionShape.RECTANGLE, points_x=points_x, points_y=points_y, labels=labels)
                    annotation.regions.append(region)

        logger.info('Finished read annotations')
        logger.info('Annotations read: {}'.format(len(annotations)))
        if skipped_annotations:
            logger.info('Annotations skipped: {}'.format(len(skipped_annotations)))

        return annotations

    def write_annotations(self, annotations, subset_type=SubsetType.TRAINVAL):
        """
        Writes YOLO annotations to the annotations folder and copy the corresponding source files.
        `annotations`: the annotations as dictionary
        `subset_type`: the subset type to write
        return: a list of written target file paths
        """
        path = join(self.path, str(subset_type))
        create_folder(path)
        annotations_path = join(path, self.annotations_folder_name)
        annotations_folder = create_folder(annotations_path)
        images_path = join(path, self.images_folder_name)
        images_folder = create_folder(images_path)
        categories = read_categories(join(self.path, self.categories_file_name))

        logger.info('Write images to {}'.format(images_folder))
        logger.info('Write annotations to {}'.format(annotations_folder))

        copied_files = []
        skipped_annotations = []

        for annotation in annotations.values():
            annotation_file_name = basename(annotation.file_path)
            file_name, _ = splitext(annotation_file_name)
            annotations_file = join(annotations_folder, '{}{}'.format(file_name, '.txt'))
            target_file = join(images_folder, annotation_file_name)

            if not isfile(annotation.file_path):
                logger.warning("{}: Source file not found, skip annotation.".format(annotation.file_path))
                skipped_annotations.append(annotation.file_path)
                continue
            if isfile(target_file):
                logger.warning("{}: Target file already exist, skip annotation.".format(annotation.file_path))
                skipped_annotations.append(annotation.file_path)
                continue

            _, image_width, image_height = get_image_size(annotation.file_path)
            rows = []
            skipped_regions = []
            for index, region in enumerate(annotation.regions):
                if region.shape != RegionShape.RECTANGLE:
                    logger.warning('Unsupported shape {}, skip region {} at path: {}'.format(region.shape,
                                                                                             index,
                                                                                             annotations_file))
                    skipped_regions.append(region)
                    continue

                x_min, x_max = region.points_x
                y_min, y_max = region.points_y
                width = x_max - x_min
                height = y_max - y_min
                # normalize bounding box
                c_x = self._normalize_value(x_min + width / 2, image_width)
                c_y = self._normalize_value(y_min + height / 2, image_height)
                width = self._normalize_value(width, image_width)
                height = self._normalize_value(height, image_height)
                label = region.labels[0] if len(region.labels) else ''
                try:
                    class_number = categories.index(label)
                except ValueError:
                    logger.warning('Unsupported label {}, skip region {} at path: {}'.format(label,
                                                                                             index,
                                                                                             annotations_file))
                    skipped_regions.append(region)
                    continue
                rows.append(dict(zip(FIELD_NAMES, [class_number, c_x, c_y, width, height])))

            if len(skipped_regions) == len(annotation.regions):
                logger.warning("{}: All regions skipped, skip annotation.".format(annotation.file_path))
                skipped_annotations.append(annotation.file_path)
                continue

            # write the annotations
            with open(annotations_file, 'w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=FIELD_NAMES, delimiter=' ')
                writer.writerows(rows)

            # copy the file
            shutil.copy2(annotation.file_path, target_file)
            copied_files.append(target_file)

        logger.info('Finished write annotations')
        logger.info('Annotations written: {}'.format(len(annotations) - len(skipped_annotations)))
        if skipped_annotations:
            logger.info('Annotations skipped: {}'.format(len(skipped_annotations)))
        return copied_files

    @classmethod
    def _denormalize_value(cls, value, metric):
        """
        Denormalize a bounding box value
        `value`: the value to denormalize
        `metric`: the metric to denormalize from
        return: the denormalized value
        """
        return int(value * metric)

    @classmethod
    def _normalize_value(cls, value, metric):
        """
        Normalize a bounding box value
        `value`: the value to normalize
        `metric`: the metric to normalize against
        return: the normalized value
        """
        return float(value) / metric