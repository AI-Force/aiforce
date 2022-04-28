import shutil
import logging
from os.path import join, splitext, basename, isfile
from ..core import assign_arg_prefix
from ..category_tools import equal_categories
from .core import Annotation, AnnotationAdapter, Region, RegionShape, SubsetType
from ..image.opencv_tools import get_image_size
from ..io.core import create_folder
import xml.etree.ElementTree as ET

# The default categories for classification and object detection
DEFAULT_CATEGORIES_MAIN = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]

# The default categories for object pose
DEFAULT_CATEGORIES_POSE = [
    'Unspecified',
    'Left',
    'Right',
    'Frontal',
    'Rear',
]

# The default person layout categories
DEFAULT_CATEGORIES_LAYOUT = [
    'head',
    'hand',
    'foot',
]

# The default categories for action classification
DEFAULT_CATEGORIES_ACTION = [
    'other',  # skip this when training classifiers
    'jumping',
    'phoning',
    'playinginstrument',
    'reading',
    'ridingbike',
    'ridinghorse',
    'running',
    'takingphoto',
    'usingcomputer',
    'walking',
]

# The available challenges in the dataset
CHALLENGES = [
    'Main',
    'Action',
    'Layout',
    'Segmentation',
]

DEFAULT_CATEGORIES = dict(zip(CHALLENGES, [DEFAULT_CATEGORIES_MAIN, DEFAULT_CATEGORIES_ACTION,
                                           DEFAULT_CATEGORIES_LAYOUT, DEFAULT_CATEGORIES_MAIN]))

# The default challenge if not specified
DEFAULT_CHALLENGE = CHALLENGES[0]
# The default object pose category if not specified
DEFAULT_POSE = DEFAULT_CATEGORIES_POSE[0]
DEFAULT_IMAGES_FOLDER = "JPEGImages"
DEFAULT_ANNOTATIONS_FOLDER = "Annotations"
DEFAULT_INDEX_FOLDER = "ImageSets"


# Cell
logger = logging.getLogger(__name__)


# Cell
class PascalVOCAnnotationAdapter(AnnotationAdapter):
    """
    Adapter to read and write annotations in the PascalVOC annotation.
    """

    def __init__(self, path, categories_file_name=None, challenge=None, index_folder_name=None,
                 images_folder_name=None, annotations_folder_name=None):
        """
        PascalVOC adapter to read and write annotations.
        `path`: the folder containing the annotations
        `categories_file_name`: the name of the categories file
        `challenge`: the challenge to use, default to Main
        `index_folder_name`: the name of the folder containing the index files, default to ImageSets
        `images_folder_name`: the name of the folder containing the index files, default to JPEGImages
        `annotations_folder_name`: the name of the folder containing the image annotations, default to Annotations
        """
        super().__init__(path, categories_file_name)

        self.challenge = DEFAULT_CHALLENGE if challenge is None else challenge
        self.index_folder_name = DEFAULT_INDEX_FOLDER if index_folder_name is None else index_folder_name
        self.images_folder_name = DEFAULT_IMAGES_FOLDER if images_folder_name is None else images_folder_name
        if annotations_folder_name is None:
            self.annotations_folder_name = DEFAULT_ANNOTATIONS_FOLDER
        else:
            self.annotations_folder_name = annotations_folder_name

    @classmethod
    def argparse(cls, prefix=None):
        """
        Returns the argument parser containing argument definition for command line use.
        `prefix`: a parameter prefix to set, if needed
        return: the argument parser
        """
        parser = super(PascalVOCAnnotationAdapter, cls).argparse(prefix=prefix)
        parser.add_argument(assign_arg_prefix('--challenge', prefix),
                            dest="challenge",
                            help="The challenge to use for read or write annotations.",
                            choices=CHALLENGES,
                            default=None)
        parser.add_argument(assign_arg_prefix('--index_folder_name', prefix),
                            dest="index_folder_name",
                            help="The name of the folder containing the index files.",
                            default=None)
        parser.add_argument(assign_arg_prefix('--images_folder_name', prefix),
                            dest="images_folder_name",
                            help="The name of the folder containing the image files.",
                            default=None)
        parser.add_argument(assign_arg_prefix('--annotations_folder_name', prefix),
                            dest="annotations_folder_name",
                            help="The name of the folder containing the image annotations.",
                            default=None)

        return parser

    def read_categories(self):
        """
        Read categories.
        return: a list of category names
        """
        categories = super(PascalVOCAnnotationAdapter, self).read_categories()
        if len(categories) == 0:
            categories = DEFAULT_CATEGORIES[self.challenge]
            logger.info(f"Use {len(categories)} default categories for the challenge {self.challenge}.")
        return categories

    def write_categories(self, categories):
        """
        Write categories.
        `categories`: a list of category names
        """
        if equal_categories(categories, DEFAULT_CATEGORIES[self.challenge]):
            logger.info(f"Default categories used for the challenge {self.challenge}. Skip write categories.")
        else:
            super(PascalVOCAnnotationAdapter, self).write_categories(categories)

    def read_annotations(self, categories, subset_type=SubsetType.TRAINVAL):
        """
        Reads Pascal VOC annotations.
        Supports all challenge formats.
        `categories`: the categories as list
        `subset_type`: the subset type to read
        return: the annotations as dictionary
        """
        index_path = join(self.path, self.index_folder_name)
        annotations_path = join(self.path, self.annotations_folder_name)
        images_path = join(self.path, self.images_folder_name)

        logger.info(f'Read index from {index_path}')
        logger.info(f'Read file sources from {images_path}')
        logger.info(f'Read annotations from {annotations_path}')
        logger.info(f'Read annotations for challenge: {self.challenge}')
        index_file_path = join(index_path, self.challenge, f"{subset_type}.txt")
        if self.challenge == 'Main':
            annotations, skipped_annotations = self._read_annotations_main(index_file_path, annotations_path,
                                                                           images_path)
        else:
            message = f'Unsupported challenge: {self.challenge}'
            logger.error(message)
            raise ValueError(message)

        logger.info(f'Finished read annotations for challenge: {self.challenge}')
        logger.info(f'Annotations read: {len(annotations)}')
        if skipped_annotations:
            logger.info(f'Annotations skipped: {len(skipped_annotations)}')

        return annotations

    def _read_annotations_main(self, index_file_path, annotations_path, images_path):
        """
        Reads Main challenge annotations.
        `index_file_path`: the path to the index file
        `annotations_path`: the path containing the annotation files
        `images_path`: the path containing the image files
        return: the annotations as dictionary
        """
        annotations = {}
        skipped_annotations = []
        with open(index_file_path) as file:
            for line in file:
                annotations_file_name = line.strip()  # preprocess line
                annotations_file_path = join(annotations_path, f"{annotations_file_name}.xml")
                tree = ET.parse(annotations_file_path)
                root = tree.getroot()
                annotation_id = root.find('filename').text
                file_path = join(images_path, annotation_id)

                if not isfile(file_path):
                    logger.warning("{}: Source file not found, skip annotation.".format(file_path))
                    skipped_annotations.append(file_path)
                    continue

                if annotation_id not in annotations:
                    annotations[annotation_id] = Annotation(annotation_id=annotation_id, file_path=file_path)

                annotation = annotations[annotation_id]

                for obj in root.iter('object'):
                    difficult = obj.find('difficult').text
                    category = obj.find('name').text

                    # if int(difficult) == 1:
                    #     continue
                    bbox = obj.find('bndbox')
                    points_x = [int(bbox.find('xmin').text), int(bbox.find('xmax').text)]
                    points_y = [int(bbox.find('ymin').text), int(bbox.find('ymax').text)]

                    labels = category.split(' ') if category else []
                    region = Region(shape=RegionShape.RECTANGLE, points_x=points_x, points_y=points_y, labels=labels)
                    annotation.regions.append(region)

        return annotations, skipped_annotations

    def write_annotations(self, annotations, categories, subset_type=SubsetType.TRAINVAL):
        """
        Writes Pascal VOC annotations.
        Supports all challenge formats.
        `annotations`: the annotations as dictionary
        `categories`: the categories as list
        `subset_type`: the subset type to write
        return: a list of written target file paths
        """
        index_path = create_folder(join(self.path, self.index_folder_name))
        annotations_path = create_folder(join(self.path, self.annotations_folder_name))
        images_path = create_folder(join(self.path, self.images_folder_name))

        logger.info(f'Write index to {index_path}')
        logger.info(f'Write file sources to {images_path}')
        logger.info(f'Write annotations to {annotations_path}')

        copied_files = set()
        logger.info(f'Write annotations for challenge: {self.challenge}')
        challenge_index_path = create_folder(join(index_path, self.challenge))
        index_file_path = join(challenge_index_path, f"{subset_type}.txt")
        if self.challenge == 'Main':
            challenge_copied_files, annotations, skipped_annotations = self._write_annotations_main(
                index_file_path, annotations_path, images_path, annotations)
        else:
            message = f'Unsupported challenge: {self.challenge}'
            logger.error(message)
            raise ValueError(message)

        copied_files.update(challenge_copied_files)

        logger.info(f'Finished write annotations for challenge: {self.challenge}')
        logger.info(f'Annotations written: {len(annotations) - len(skipped_annotations)}')
        if skipped_annotations:
            logger.info(f'Annotations skipped: {len(skipped_annotations)}')

        return list(copied_files)

    def _write_annotations_main(self, index_file_path, annotations_path, images_path, annotations):
        """
        Writes Main challenge annotations and copy the corresponding source files.
        `index_file_path`: the path to the index file
        `annotations_path`: the path to write annotation files into
        `images_path`: the path to write image files into
        `annotations`: the annotations to write
        return: a list of written target file paths
        """
        index_list = []
        skipped_annotations = []
        copied_files = []
        for annotation in annotations.values():
            image_name = basename(annotation.file_path)
            file_name, _ = splitext(image_name)
            annotations_file = join(annotations_path, f'{file_name}.xml')
            target_file = join(images_path, image_name)

            if not isfile(annotation.file_path):
                logger.warning("{}: Source file not found, skip annotation.".format(annotation.file_path))
                skipped_annotations.append(annotation.file_path)
                continue
            if isfile(target_file):
                logger.warning("{}: Target file already exist, skip annotation.".format(annotation.file_path))
                skipped_annotations.append(annotation.file_path)
                continue

            img, image_width, image_height = get_image_size(annotation.file_path)

            root = ET.Element('annotation')
            ET.SubElement(root, "filename").text = image_name
            ET.SubElement(root, "folder").text = basename(self.path)
            size = ET.SubElement(root, "size")
            ET.SubElement(size, "depth").text = str(img.shape[-1]) if len(img.shape) == 3 else '1'
            ET.SubElement(size, "width").text = str(image_width)
            ET.SubElement(size, "height").text = str(image_height)

            skipped_regions = []
            for index, region in enumerate(annotation.regions):
                if region.shape != RegionShape.RECTANGLE:
                    logger.warning('Unsupported shape {}, skip region {} at path: {}'.format(region.shape,
                                                                                             index,
                                                                                             annotations_file))
                    skipped_regions.append(region)
                    continue

                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = ' '.join(region.labels) if len(region.labels) else ''
                bbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bbox, "xmin").text = str(region.points_x[0])
                ET.SubElement(bbox, "xmax").text = str(region.points_x[1])
                ET.SubElement(bbox, "ymin").text = str(region.points_y[0])
                ET.SubElement(bbox, "ymax").text = str(region.points_y[1])
                ET.SubElement(obj, "difficult").text = '0'
                ET.SubElement(obj, "pose").text = DEFAULT_POSE

            if len(skipped_regions) == len(annotation.regions):
                logger.warning("{}: All regions skipped, skip annotation.".format(annotation.file_path))
                skipped_annotations.append(annotation.file_path)
                continue

            with open(annotations_file, 'w') as f:
                ET.ElementTree(root).write(f, encoding='unicode')

            # copy the file
            shutil.copy2(annotation.file_path, target_file)
            copied_files.append(target_file)
            index_list.append(file_name)

        with open(index_file_path, 'w') as file:
            file.write("\n".join(index_list))

        return copied_files, annotations, skipped_annotations
