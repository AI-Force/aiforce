# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/dataset-image_segmentation.ipynb (unless otherwise specified).

__all__ = ['logger', 'ImageSegmentationDataset']

# Cell

import numpy as np
import logging
from os.path import join, isfile, splitext, basename
from functools import partial
from ..core import assign_arg_prefix, input_feedback
from ..annotation.core import AnnotationAdapter
from .image_object_detection import ImageObjectDetectionDataset
from ..image.pillow_tools import assign_exif_orientation, get_image_size, write_mask
from ..io.core import create_folder
from ..annotation.core import RegionShape, convert_region, region_bounding_box

# Cell

logger = logging.getLogger(__name__)

# Cell


class ImageSegmentationDataset(ImageObjectDetectionDataset):

    SEMANTIC_MASK_FOLDER = 'semantic_masks'

    def __init__(self, input_adapter: AnnotationAdapter, output_adapter: AnnotationAdapter, split=None, seed=None,
                 sample=None, tfrecord=False, join_overlapping_regions=False, annotation_area_threshold=None,
                 generate_semantic_masks=True):
        super().__init__(input_adapter, output_adapter, split, seed, sample, tfrecord, join_overlapping_regions,
                         annotation_area_threshold)
        self.generate_semantic_masks = generate_semantic_masks
        self.semantic_mask_folder = join(self.output_adapter.path, self.SEMANTIC_MASK_FOLDER)

    @classmethod
    def argparse(cls, prefix=None):
        """
        Returns the argument parser containing argument definition for command line use.
        `prefix`: a parameter prefix to set, if needed
        return: the argument parser
        """
        parser = super(ImageSegmentationDataset, cls).argparse(prefix=prefix)
        parser.add_argument(assign_arg_prefix("--generate_semantic_masks", prefix),
                            dest="generate_semantic_masks",
                            help="Whether semantic masks should be generated.",
                            action="store_true",
                            default=True)
        return parser

    def create_folders(self):
        """
        Creates the data-set folder structure, if not exist
        """
        super().create_folders()

        if self.generate_semantic_masks:
            # create semantic mask file folder and remove previous data if exist
            semantic_mask_folder = create_folder(self.semantic_mask_folder, clear=True)
            logger.info("Created semantic mask folder {}".format(semantic_mask_folder))

    def copy(self, train_annotation_keys, val_annotation_keys, test_files=None):
        """
        Copy the images to the dataset and remove EXIF orientation information by hard-rotate the images.
        If tfrecords should be build, create tfrecords for train and val subsets and generate a labelmap.pbtxt file.
        If semantic masks should be generate, masks for train and val subsets are build.
        `train_annotation_keys`: The list of training annotation keys
        `val_annotation_keys`: The list of validation annotation keys
        `test_files`: The list of test file paths
        return: A tuple containing train, val and test target file paths
        """

        train_targets, val_targets, test_targets = super().copy(train_annotation_keys, val_annotation_keys, test_files)

        if self.generate_semantic_masks:
            # save semantic masks
            self._save_semantic_masks(train_annotation_keys + val_annotation_keys)

        return train_targets, val_targets, test_targets

    def convert_annotations(self):
        """
        Converts segmentation regions from rectangle to polygon, if exist
        """

        # only the trainval images have annotation, not the test images
        steps = [
            {
                'name': 'position',
                'choices': {
                    's': 'Skip',  # just delete the annotation
                    'S': 'Skip All',
                    't': 'Trim',  # transform the annotation
                    'T': 'Trim All',
                },
                'choice': None,
                'condition': lambda p_min, p_max, size: p_min < 0 or p_max >= size,
                'message': '{} -> {} : {}Exceeds image {}. \n Points \n x: {} \n y: {}',
                'transform': lambda p, size=0: max(min(p, size - 1), 0),
            },
            {
                'name': 'size',
                'choices': {
                    's': 'Skip',  # just delete the annotation
                    'S': 'Skip All',
                    'k': 'Keep',  # transform the annotation (in this case do nothing)
                    'K': 'Keep All',
                },
                'choice': None,
                'condition': lambda p_min, p_max, _: p_max - p_min <= 1,
                'message': '{} -> {} : {}Shape {} is <= 1 pixel. \n Points \n x: {} \n y: {}',
                'transform': lambda p, size=0: p,
            }
        ]

        logger.info('Start convert image annotations from {}'.format(self.input_adapter.path))

        for annotation in self.annotations.values():
            # skip file, if regions are empty or file do not exist
            if not (annotation.regions and isfile(annotation.file_path)):
                continue

            image, _, __ = assign_exif_orientation(annotation.file_path)
            image_width, image_height = image.size

            delete_regions = {}
            for index, region in enumerate(annotation.regions):
                # convert from rect to polygon if needed
                convert_region(region, RegionShape.POLYGON)

                for step in steps:
                    # validate the shape size
                    (x_min, x_max), (y_min, y_max) = region_bounding_box(region)

                    width_condition = step['condition'](x_min, x_max, image_width)
                    height_condition = step['condition'](y_min, y_max, image_height)
                    if width_condition or height_condition:
                        size_message = ['width'] if width_condition else []
                        size_message.extend(['height'] if height_condition else [])
                        message = step['message'].format(annotation.file_path, index, ' ', ' and '.join(size_message),
                                                         region.points_x, region.points_y)

                        step['choice'] = input_feedback(message, step['choice'], step['choices'])

                        choice_op = step['choice'].lower()
                        # if skip the shapes
                        if choice_op == 's':
                            delete_regions[index] = True
                            message = step['message'].format(annotation.file_path, index,
                                                             '{} '.format(step['choices'][choice_op]),
                                                             ' and '.join(size_message),
                                                             region.points_x, region.points_y)
                            logger.info(message)

                            break
                        else:
                            region.points_x = list(map(partial(step['transform'], size=image_width), region.points_x))
                            region.points_y = list(map(partial(step['transform'], size=image_height), region.points_y))

                            message = step['message'].format(annotation.file_path, index,
                                                             '{} '.format(step['choices'][choice_op]),
                                                             ' and '.join(size_message),
                                                             region.points_x, region.points_y)
                            logger.info(message)

            # delete regions after iteration is finished
            for index in sorted(list(delete_regions.keys()), reverse=True):
                del annotation.regions[index]

        print('Finished convert image annotations from {}'.format(self.input_adapter.path))

    def _save_semantic_masks(self, annotation_keys):
        """
        Create semantic segmentation mask png files out of the annotations.
        The mask file name is the same as the image file name but is stored in png format.
        `annotation_keys`: The annotation keys to create the segmentation masks for
        """
        from skimage import draw

        num_masks = len(annotation_keys)
        logger.info('Start create {} segmentation masks in {}'.format(num_masks, self.semantic_mask_folder))

        # only the trainval images have annotation, not the test images
        for index, key in enumerate(annotation_keys):
            annotation = self.annotations[key]

            if not annotation.regions:
                continue

            image, image_width, image_height = get_image_size(annotation.file_path)

            # Convert polygons to a bitmap mask of shape
            # [height, width]
            mask = np.zeros((image_height, image_width), dtype=np.uint8)

            # sort the regions by category priority for handling pixels which are assigned to more than one category
            # the category with higher index paint over the category with lower index
            for region in sorted(annotation.regions, key=lambda r: self.categories.index(r.labels[0])):
                class_id = self.categories.index(region.labels[0]) + 1

                # Get indexes of pixels inside the polygon and set them to 1
                rr, cc = draw.polygon(region.points_y, region.points_x)
                mask[rr, cc] = class_id

            # save the semantic mask
            file_name = basename(annotation.file_path)
            mask_path = join(self.semantic_mask_folder, splitext(file_name)[0] + '.png')
            write_mask(mask, mask_path)

            logger.info('{} / {} - Created segmentation mask {}'.format(index + 1, num_masks, mask_path))

        logger.info('Finish create {} segmentation masks in {}'.format(num_masks, self.semantic_mask_folder))
