# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/dataset-image_classification.ipynb (unless otherwise specified).

__all__ = ['logger', 'ImageClassificationDataset']

# Cell

import logging
from os.path import join
from ..core import assign_arg_prefix
from ..annotation.core import AnnotationAdapter
from .core import Dataset
from ..tensorflow.tfrecord_builder import create_labelmap_file

# Cell

logger = logging.getLogger(__name__)

# Cell


class ImageClassificationDataset(Dataset):
    """
    Classification dataset.
    """

    def __init__(self, input_adapter: AnnotationAdapter, output_adapter: AnnotationAdapter, split=None, seed=None,
                 sample=None, tfrecord=False):
        super().__init__(input_adapter, output_adapter, split, seed, sample)
        self.tfrecord = tfrecord

    @classmethod
    def argparse(cls, prefix=None):
        """
        Returns the argument parser containing argument definition for command line use.
        `prefix`: a parameter prefix to set, if needed
        return: the argument parser
        """
        parser = super(ImageClassificationDataset, cls).argparse(prefix=prefix)
        parser.add_argument(assign_arg_prefix("--tfrecord", prefix),
                            dest="tfrecord",
                            help="Also create .tfrecord files.",
                            action="store_true")
        return parser

    def copy(self, train_annotation_keys, val_annotation_keys, test_files=None):
        """
        Copy the images to the dataset and remove EXIF orientation information by hard-rotate the images.
        If tfrecords are build, generate a labelmap.pbtxt file.
        `train_annotation_keys`: The list of training annotation keys
        `val_annotation_keys`: The list of validation annotation keys
        `test_files`: The list of test file paths
        return: A tuple containing train, val and test target file paths
        """

        train_targets, val_targets, test_targets = super().copy(train_annotation_keys, val_annotation_keys, test_files)

        files = train_targets + val_targets + test_targets
        logger.info('Start assign image orientation to {} images'.format(len(files)))
        for file in files:
            self.assign_orientation(file)
        logger.info('Finished assign image orientation to {} images'.format(len(files)))

        # if create tfrecord, create a labelmap.pbtxt file containing the categories
        if self.tfrecord:
            labelmap_file_name = 'label_map.pbtxt'
            labelmap_output_file = join(self.output_adapter.path, labelmap_file_name)
            logger.info('Generate {}'.format(labelmap_output_file))
            create_labelmap_file(labelmap_output_file, list(self.categories), 1)

        return train_targets, val_targets, test_targets

    def build_info(self):
        """
        Converts annotations
        """
        super().build_info()
        logger.info('create_tfrecord: {}'.format(self.tfrecord))
