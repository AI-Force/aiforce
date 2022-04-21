import argparse
import logging
import random
import numpy as np
from abc import ABC
from os.path import join, basename, dirname
from ..core import assign_arg_prefix
from ..annotation.core import AnnotationAdapter, SubsetType
from ..image.pillow_tools import assign_exif_orientation, write_exif_metadata
from ..io.core import create_folder


logger = logging.getLogger(__name__)


class Dataset(ABC):
    """
    Dataset base class to build datasets.
    `args`: the arguments containing the parameters
    """

    DEFAULT_SPLIT = 0.2

    def __init__(self, input_adapter: AnnotationAdapter, output_adapter: AnnotationAdapter, split=None, seed=None,
                 sample=None):
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter
        self.split = self.DEFAULT_SPLIT if split is None else split
        self.seed = seed
        self.sample = sample
        self.categories = input_adapter.read_categories()
        self.annotations = input_adapter.read_annotations()

    @classmethod
    def argparse(cls, prefix=None):
        """
        Returns the argument parser containing argument definition for command line use.
        `prefix`: a parameter prefix to set, if needed
        return: the argument parser
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(assign_arg_prefix('--split', prefix),
                            dest="split",
                            help="Percentage of the data which belongs to validation set.",
                            type=float,
                            default=0.2)
        parser.add_argument(assign_arg_prefix('--seed', prefix),
                            dest="seed",
                            help="A random seed to reproduce splits.",
                            type=int,
                            default=None)
        parser.add_argument(assign_arg_prefix('--sample', prefix),
                            dest="sample",
                            help="Percentage of the data which will be copied as a sample set.",
                            type=float,
                            default=0)

        return parser

    def create_folders(self):
        """
        Creates the dataset folder structure, if not exist
        """
        output_folder = create_folder(self.output_adapter.path, clear=True)
        logger.info("Created folder {}".format(output_folder))

    def build_info(self):
        """
        Log build information
        """
        logger.info('Build configuration:')
        logger.info('input_adapter: {}'.format(type(self.input_adapter).__name__))
        logger.info('input_path: {}'.format(self.input_adapter.path))
        logger.info('output_adapter: {}'.format(type(self.output_adapter).__name__))
        logger.info('output_path: {}'.format(self.output_adapter.path))
        logger.info('split: {}'.format(self.split))
        logger.info('seed: {}'.format(self.seed))
        logger.info('sample: {}'.format(self.sample))

    def validate(self):
        """
        Validates the annotations.
        return: The skipped annotations
        """
        return {}

    def copy(self, train_annotation_keys, val_annotation_keys, test_files=None):
        """
        Copy the images to the dataset.
        `train_annotation_keys`: The list of training annotation keys
        `val_annotation_keys`: The list of validation annotation keys
        `test_files`: The list of test file paths
        return: A tuple containing train, val and test target file paths
        """

        logger.info('Start copy annotations from {} to {}'.format(self.input_adapter.path,
                                                                  self.output_adapter.path))

        # copy the categories files
        logger.info('Write categories to {}'.format(self.output_adapter.path))
        self.output_adapter.write_categories(self.categories)

        logger.info('Write {} annotations to {}'.format(str(SubsetType.TRAIN), self.output_adapter.path))
        annotations_train = dict(zip(train_annotation_keys, [self.annotations[key] for key in train_annotation_keys]))
        train_targets = self.output_adapter.write_annotations(annotations_train, SubsetType.TRAIN)
        logger.info('Write {} annotations to {}'.format(str(SubsetType.VAL), self.output_adapter.path))
        annotations_val = dict(zip(val_annotation_keys, [self.annotations[key] for key in val_annotation_keys]))
        val_targets = self.output_adapter.write_annotations(annotations_val, SubsetType.VAL)
        logger.info('Write {} files to {}'.format(str(SubsetType.TEST), self.output_adapter.path))
        test_targets = self.output_adapter.write_files(test_files, SubsetType.TEST) if test_files else []

        return train_targets, val_targets, test_targets

    def build(self, validate=True):
        """
        Build the data-set. This is the main logic.
        This method validates the images against the annotations,
        split the image-set into train and val on given split percentage,
        creates the data-set folders and copies the image.
        If a sample percentage is given, a sub-set is created as sample.
        `validate`: True if annotations should be validate, else False
        """
        logger.info('Validation set contains {}% of the images.'.format(int(self.split * 100)))

        # validate the image set
        skipped_annotations = self.validate() if validate else {}
        if len(skipped_annotations) > 0:
            logger.info(f"Validation finished, skipped {len(skipped_annotations)} annotations.")

        # split category files into train & val and create the sample split, if set
        train_annotation_keys = []
        val_annotation_keys = []
        sample_train_annotation_keys = []
        sample_val_annotation_keys = []

        train, val = self.split_train_val_data(list(self.annotations.keys()), self.split, self.seed)

        train_annotation_keys.extend(train)
        val_annotation_keys.extend(val)
        # if test files exist
        test_files = self.input_adapter.list_files(SubsetType.TEST)

        # if a sample data set should be created, create the splits
        if self.sample:
            _, sample_train = self.split_train_val_data(train, self.sample, self.seed)
            _, sample_val = self.split_train_val_data(val, self.sample, self.seed)
            sample_train_annotation_keys.extend(sample_train)
            sample_val_annotation_keys.extend(sample_val)
            # if test files exist
            if test_files:
                _, sample_test_files = self.split_train_val_data(test_files, self.sample, self.seed)
            else:
                sample_test_files = None

        # copy the annotations
        self.copy(train_annotation_keys, val_annotation_keys, test_files)

        if self.sample:
            # backup original output path
            output_path = self.output_adapter.path
            sample_name = "{}_sample".format(basename(output_path))
            # set output path to sample set
            self.output_adapter.path = join(dirname(output_path), sample_name)
            logger.info('Start build {} dataset containing {}% of images at {}'.format(sample_name,
                                                                                       int(self.sample * 100),
                                                                                       self.output_adapter.path))
            # create the sample data set folder
            self.create_folders()
            # copy the sample data
            self.copy(sample_train_annotation_keys, sample_val_annotation_keys, sample_test_files)

            logger.info('Finished build {} dataset containing {}% of images at {}'.format(sample_name,
                                                                                          int(self.sample * 100),
                                                                                          self.output_adapter.path))
            # restore original output path
            self.output_adapter.path = output_path

    @classmethod
    def split_train_val_data(cls, data, val_size=0.1, seed=None):
        """
        Splits the images in train and validation set
        `data`: the data to split
        `val_size`: the size of the validation set in percentage
        `seed`: A random seed to reproduce splits.
        return: the split train, validation images
        """
        data_len = len(data)
        if data_len <= 1 or val_size == 0:
            return data, []
        elif val_size == 1:
            return [], data

        random.seed(seed)
        indices = random.sample(range(len(data)), k=int(val_size * data_len))
        data = np.array(data)
        val = data[indices]
        train = np.delete(data, indices)
        return train.tolist(), val.tolist()

    @classmethod
    def assign_orientation(cls, file_path):
        """
        Assign the EXIF metadata orientation to an image.
        `file_path`: the path to the image file
        """

        # rotate image by EXIF orientation metadata and remove them
        image, exif_data, rotated = assign_exif_orientation(file_path)
        if rotated:
            write_exif_metadata(image, exif_data, file_path)


class ImageDataset(Dataset):
    """
    Abstract class for Dataset handling images.
    Used as a base class for Computer Vision tasks.
    """

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

        return train_targets, val_targets, test_targets
