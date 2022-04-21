import logging
from os.path import join
from ..core import assign_arg_prefix
from ..annotation.core import AnnotationAdapter
from .core import ImageDataset
from ..tensorflow.tfrecord_builder import create_labelmap_file


logger = logging.getLogger(__name__)


class ImageClassificationDataset(ImageDataset):
    """
    Image Classification dataset.
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

    def validate(self):
        """
        Validates the annotations.
        return: The skipped annotations
        """
        # validate only the trainval images, the test images have no annotations to validate
        logger.info('Start validate data at {}'.format(self.input_adapter.path))

        files = self.input_adapter.list_files()

        logger.info('Found {} files at {}'.format(len(files), self.input_adapter.path))

        delete_annotations = {}
        used_categories = set([])

        for annotation_id, annotation in self.annotations.items():

            delete_regions = {}
            for index, region in enumerate(annotation.regions):
                len_labels = len(region.labels)
                region_valid = len_labels and len(set(region.labels) & set(self.categories)) == len_labels
                if not region_valid:
                    message = '{} : Region {} with category {} is not in category list, skip region.'
                    logger.info(message.format(annotation.file_path, index, ','.join(region.labels)))

                    delete_regions[index] = True
                else:
                    # update the used regions
                    used_categories.update(region.labels)

            # delete regions after iteration is finished
            for index in sorted(list(delete_regions.keys()), reverse=True):
                del annotation.regions[index]

            # validate for empty region
            if not annotation.regions:
                logger.info('{} : Has empty regions, skip annotation.'.format(annotation.file_path))
                delete_annotations[annotation_id] = True
            # validate for file exist
            elif annotation.file_path not in files:
                logger.info('{} : File of annotations do not exist, skip annotations.'.format(annotation.file_path))
                delete_annotations[annotation_id] = True
            else:
                files.pop(files.index(annotation.file_path))

        for index, file in enumerate(files):
            logger.info('[{}] -> {} : File has no annotations, skip file.'.format(index, file))

        # list unused categories
        empty_categories = frozenset(self.categories) - used_categories
        if empty_categories:
            logger.info('The following categories have no images: {}'.format(" , ".join(empty_categories)))

        # delete annotations after iteration is finished
        for index in delete_annotations.keys():
            del self.annotations[index]

        logger.info('Finished validate image set at {}'.format(self.input_adapter.path))
        return delete_annotations
