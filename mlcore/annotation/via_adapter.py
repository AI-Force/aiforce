# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/annotation-via_adapter.ipynb (unless otherwise specified).

__all__ = ['DEFAULT_ANNOTATIONS_FILE', 'DEFAULT_CATEGORY_ID', 'CSV_FIELDNAMES', 'logger', 'VIAAdapter',
           'configure_logging']

# Cell

import json
import csv
import sys
import shutil
import argparse
import logging
from os.path import join, splitext, getsize, basename, dirname, isfile
from .core import Annotation, AnnotationAdapter, Region, RegionShape, parse_region_shape
from ..io.core import create_folder

# Cell

DEFAULT_ANNOTATIONS_FILE = 'via_region_data.json'
DEFAULT_CATEGORY_ID = 'category'
CSV_FIELDNAMES = ['#filename', 'file_size', 'file_attributes', 'region_count', 'region_id', 'region_shape_attributes',
                  'region_attributes']

# Cell

logger = logging.getLogger(__name__)

# Cell


class VIAAdapter(AnnotationAdapter):
    """
    Adapter to read and write annotations in the VIA annotation.
    `args`: the arguments containing the parameters
    """

    def __init__(self, args):
        super().__init__()
        self.files_path = args.files_path
        self.annotations_file = args.annotations_file
        self.category_label_key = DEFAULT_CATEGORY_ID if args.category_label_key is None else args.category_label_key

    def read(self):
        """
        Reads a VIA annotations file.
        Supports JSON and CSV file format.
        return: the annotations as dictionary
        """
        logger.info('Read annotations from {}'.format(self.annotations_file))

        return self._read_v1()

    def _read_v1(self):
        """
        Reads a VIA v1 annotations file.
        Supports JSON and CSV file format.
        return: the annotations as dictionary
        """
        file_extension = splitext(self.annotations_file)[1]

        if file_extension.lower() == '.json':
            logger.info('Read VIA v1 annotations in JSON format')
            annotations = self._read_v1_json()
        elif file_extension.lower() == '.csv':
            logger.info('Read VIA v1 annotations in CSV format')
            annotations = self._read_v1_csv()
        else:
            message = 'Unsupported annotation format at {}'.format(self.annotations_file)
            logger.error(message)
            raise ValueError(message)

        return annotations

    def _read_v1_csv(self):
        """
        Reads a VIA v1 CSV annotations file.
        return: the annotations as dictionary
        """
        annotations = {}

        with open(self.annotations_file, newline='') as csv_file:
            reader = csv.DictReader(csv_file)

            skipped_annotations = []
            for row in reader:
                file_path = join(self.files_path, row['#filename'])
                if not isfile(file_path):
                    logger.warning("{}: Source file not found, skip annotation.".format(file_path))
                    skipped_annotations.append(file_path)
                    continue

                annotation_id = "{}{}".format(row['#filename'], row['file_size'])

                if annotation_id not in annotations:
                    annotations[annotation_id] = Annotation(annotation_id=annotation_id, file_path=file_path)

                annotation = annotations[file_path]

                region_shape_attributes = json.loads(row['region_shape_attributes'])
                region = self._parse_region_shape_attributes(region_shape_attributes)
                region_attributes = json.loads(row['region_attributes'])
                category = None
                if region_attributes and self.category_label_key in region_attributes:
                    category = region_attributes[self.category_label_key]
                region.labels = [category] if category else []
                annotation.regions.append(region)

        logger.info('Finished read annotations')
        logger.info('Annotations read: {}'.format(len(annotations)))
        if skipped_annotations:
            logger.info('Annotations skipped: {}'.format(len(skipped_annotations)))

        return annotations

    def _read_v1_json(self):
        """
        Reads a VIA v1 JSON annotations file.
        return: the annotations as dictionary
        """
        annotations = {}

        with open(self.annotations_file) as json_file:
            via_annotations = json.load(json_file)

            skipped_annotations = []
            for data in via_annotations.values():
                file_path = join(self.files_path, data['filename'])
                if not isfile(file_path):
                    logger.warning("{}: Source file not found, skip annotation.".format(file_path))
                    skipped_annotations.append(file_path)
                    continue

                annotation_id = "{}{}".format(data['filename'], data['size'])

                if annotation_id not in annotations:
                    annotations[annotation_id] = Annotation(annotation_id=annotation_id, file_path=file_path)

                annotation = annotations[annotation_id]

                for region_data in data['regions'].values():
                    region_shape_attributes = region_data['shape_attributes']
                    region = self._parse_region_shape_attributes(region_shape_attributes)
                    region_attributes = region_data['region_attributes']
                    category = None
                    if region_attributes and self.category_label_key in region_attributes:
                        category = region_attributes[self.category_label_key]
                    region.labels = [category] if category else []
                    annotation.regions.append(annotation)

        logger.info('Finished read annotations')
        logger.info('Annotations read: {}'.format(len(annotations)))
        if skipped_annotations:
            logger.info('Annotations skipped: {}'.format(len(skipped_annotations)))

        return annotations

    def write(self, annotations):
        """
        Writes a VIA annotations file and copy the corresponding source files.
        Supports JSON and CSV file format. Format is inferred from the annotations_file setting.
        `annotations`: the annotations to write
        """
        target_folder = create_folder(self.files_path)
        create_folder(dirname(self.annotations_file))
        logger.info('Write annotations to {}'.format(self.annotations_file))
        logger.info('Write file sources to {}'.format(target_folder))

        self._write_v1(annotations)

    def _write_v1(self, annotations):
        """
        Writes a VIA v1 annotations file and copy the corresponding source files.
        Supports JSON and CSV file format.
        `annotations`: the annotations to write
        """
        file_extension = splitext(self.annotations_file)[1]

        if file_extension.lower() == '.json':
            logger.info('Write VIA v1 annotations in JSON format')
            self._write_v1_json(annotations)
        elif file_extension.lower() == '.csv':
            logger.info('Write VIA v1 annotations in CSV format')
            self._write_v1_csv(annotations)
        else:
            message = 'Unsupported annotation format at {}'.format(self.annotations_file)
            logger.error(message)
            raise ValueError(message)

    def _write_v1_csv(self, annotations):
        """
        Writes a VIA v1 CSV annotations file and copy the corresponding source files.
        `annotations`: the annotations to write
        """
        with open(self.annotations_file, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()

            skipped_annotations = []

            for annotation in annotations.values():
                target_file = join(self.files_path, basename(annotation.file_path))

                if not isfile(annotation.file_path):
                    logger.warning("{}: Source file not found, skip annotation.".format(annotation.file_path))
                    skipped_annotations.append(annotation.file_path)
                    continue
                if isfile(target_file):
                    logger.warning("{}: Target file already exist, skip annotation.".format(annotation.file_path))
                    skipped_annotations.append(annotation.file_path)
                    continue

                file_size = getsize(annotation.file_path)
                file_name = basename(annotation.file_path)
                for index, region in enumerate(annotation.regions):
                    region_shape_attributes = self._create_region_shape_attributes(region)
                    region_attributes = {
                        self.category_label_key: ' '.join(region.labels) if len(region.labels) else ''
                    }

                    writer.writerow({'#filename': file_name,
                                     'file_size': file_size,
                                     'file_attributes': '{}',
                                     'region_count': len(annotation.regions),
                                     'region_id': str(index),
                                     'region_shape_attributes': json.dumps(region_shape_attributes),
                                     'region_attributes': json.dumps(region_attributes)})
                # copy the file
                shutil.copy2(annotation.file_path, target_file)

        logger.info('Finished write annotations')
        logger.info('Annotations written: {}'.format(len(annotations) - len(skipped_annotations)))
        if skipped_annotations:
            logger.info('Annotations skipped: {}'.format(len(skipped_annotations)))

    def _write_v1_json(self, annotations):
        """
        Writes a VIA v1 JSON annotations file and copy the corresponding source files.
        `annotations`: the annotations to write
        """
        json_annotations = {}
        skipped_annotations = []

        for annotation in annotations.values():
            target_file = join(self.files_path, basename(annotation.file_path))

            if not isfile(annotation.file_path):
                logger.warning("{}: Source file not found, skip annotation.".format(annotation.file_path))
                skipped_annotations.append(annotation.file_path)
                continue
            if isfile(target_file):
                logger.warning("{}: Target file already exist, skip annotation.".format(annotation.file_path))
                skipped_annotations.append(annotation.file_path)
                continue

            file_size = getsize(annotation.file_path)
            file_name = basename(annotation.file_path)
            file_id = '{:s}{:d}'.format(file_name, file_size)
            regions = {}
            for index, region in enumerate(annotation.regions):
                regions[str(index)] = {
                    'shape_attributes': self._create_region_shape_attributes(region),
                    'region_attributes': {
                        self.category_label_key: ' '.join(region.labels) if len(region.labels) else ''
                    }
                }
            json_annotations[file_id] = {
                'fileref': "",
                'size': file_size,
                'filename': file_name,
                'base64_img_data': "",
                'file_attributes': '{}',
                "regions": regions
            }
            # copy the file
            shutil.copy2(annotation.file_path, target_file)

        with open(self.annotations_file, 'w') as json_file:
            json.dump(json_annotations, json_file)

        logger.info('Finished write annotations')
        logger.info('Annotations written: {}'.format(len(annotations) - len(skipped_annotations)))
        if skipped_annotations:
            logger.info('Annotations skipped: {}'.format(len(skipped_annotations)))

    @classmethod
    def _parse_region_shape_attributes(cls, region_shape_attributes):
        """
        Parse region shape attributes.
        `region_shape_attributes`: the region shape attributes as dictionary
        return: the corresponding annotation
        """
        if not region_shape_attributes:
            return Annotation()

        region_shape = parse_region_shape(region_shape_attributes['name'])
        points_x = None
        points_y = None
        radius_x = 0
        radius_y = 0
        if region_shape == RegionShape.CIRCLE:
            points_x = [region_shape_attributes['cx']]
            points_y = [region_shape_attributes['cy']]
            radius_x = region_shape_attributes['r']
            radius_y = region_shape_attributes['r']
        elif region_shape == RegionShape.ELLIPSE:
            points_x = [region_shape_attributes['cx']]
            points_y = [region_shape_attributes['cy']]
            radius_x = region_shape_attributes['rx']
            radius_y = region_shape_attributes['ry']
        elif region_shape == RegionShape.POINT:
            points_x = [region_shape_attributes['cx']]
            points_y = [region_shape_attributes['cy']]
        elif region_shape == RegionShape.POLYGON:
            points_x = region_shape_attributes['all_points_x']
            points_y = region_shape_attributes['all_points_y']
        elif region_shape == RegionShape.RECTANGLE:
            x = region_shape_attributes['x']
            y = region_shape_attributes['y']
            width = region_shape_attributes['width']
            height = region_shape_attributes['height']
            points_x = [x, x + width]
            points_y = [y, y + height]
        return Region(shape=region_shape, points_x=points_x, points_y=points_y, radius_x=radius_x, radius_y=radius_y)

    @classmethod
    def _create_region_shape_attributes(cls, region: Region):
        """
        Create region shape attributes.
        `region`: the region to create region shape attributes from
        return: the corresponding region shape attributes as dictionary
        """
        region_shape_attributes = {
            "name": str(region.shape),

        }
        c_x = region.points_x[0] if len(region.points_x) else 0
        c_y = region.points_y[0] if len(region.points_y) else 0

        if region.shape == RegionShape.CIRCLE:
            region_shape_attributes['cx'] = c_x
            region_shape_attributes['cy'] = c_y
            region_shape_attributes['r'] = max(region.radius_x, region.radius_y)
        elif region.shape == RegionShape.ELLIPSE:
            region_shape_attributes['cx'] = c_x
            region_shape_attributes['cy'] = c_y
            region_shape_attributes['rx'] = region.radius_x
            region_shape_attributes['ry'] = region.radius_y
        elif region.shape == RegionShape.POINT:
            region_shape_attributes['cx'] = c_x
            region_shape_attributes['cy'] = c_y
        elif region.shape == RegionShape.POLYGON:
            region_shape_attributes['all_points_x'] = region.points_x
            region_shape_attributes['all_points_y'] = region.points_y
        elif region.shape == RegionShape.RECTANGLE:
            region_shape_attributes['x'] = region.points_x[0]
            region_shape_attributes['y'] = region.points_y[0]
            region_shape_attributes['width'] = region.points_x[1] - region.points_x[0]
            region_shape_attributes['height'] = region.points_y[1] - region.points_y[0]
        return region_shape_attributes

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
                            help="The path to the VIA annotation file.",
                            required=True)
        parser.add_argument(cls.assign_prefix('--category_label_key', prefix),
                            dest="category_label_key",
                            help="The key of the category label.",
                            default=DEFAULT_CATEGORY_ID)

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
