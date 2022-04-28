
# Cell
import argparse
import logging
import os
from .dataset.type import DatasetType


# Cell
DEFAULT_CATEGORIES_FILE = 'categories.txt'
NOT_CATEGORIZED = '[NOT_CATEGORIZED]'
BACKGROUND_CLASS = '_background_'
BACKGROUND_CLASS_CODE = 0


# Cell

logger = logging.getLogger(__name__)


def equal_categories(left, right):
    """
    Verify if two categories are the same
    left: the left categories to verify
    right: the right categories to verify against
    return: True if both category have exact the same contents else False
    """
    return len(left) == len(right) and labels_in_categories(left, right)


def labels_in_categories(labels, categories):
    """
    Verify if labels are all in the categories
    labels: the labels to verify
    categories: the categories to verify against
    return: True if labels in categories else False
    """
    len_labels = len(labels)
    return len_labels and len(set(labels) & set(categories)) == len_labels


# Cell
def read_categories(categories_file=None, dataset_type=DatasetType.IMAGE_CLASSIFICATION):
    """
    Reads the categories from a categories file.
    If the dataset type is image segmentation or object detection, a background class at index 0 is prepend.
    If the optional `categories_file` is not given, the file name *categories.txt* is used by default
    `categories_file`: the categories file name, if not the default
    `dataset_type`: the type of the data-set to create the categories for
    return: a list of the category names
    """
    if categories_file is None:
        categories_file = DEFAULT_CATEGORIES_FILE

    if not os.path.isfile(categories_file):
        logger.warning('Categories file not found at: {}'.format(categories_file))
        return []
    with open(categories_file) as f:
        categories = f.read().strip().split('\n')
        logger.info('Read {} categories from categories file at: {}'.format(len(categories), categories_file))
    if dataset_type in [DatasetType.IMAGE_OBJECT_DETECTION, DatasetType.IMAGE_SEGMENTATION]:
        categories = [BACKGROUND_CLASS] + categories
        logger.info('Prepend background class {} to the categories'.format(BACKGROUND_CLASS))

    return categories


# Cell
def write_categories(categories, categories_file=None):
    """
    Write the categories to a categories file.
    If the dataset type is image segmentation or object detection, a background class at index 0 is prepend.
    If the optional `categories_file` is not given, the file name *categories.txt* is used by default
    `categories`: a list of the category names to write
    `categories_file`: the categories file name
    """
    if categories_file is None:
        categories_file = DEFAULT_CATEGORIES_FILE

    if len(categories) > BACKGROUND_CLASS_CODE and categories[BACKGROUND_CLASS_CODE] == BACKGROUND_CLASS:
        logger.info('Remove background class {} from the categories'.format(BACKGROUND_CLASS))
        categories = categories[1:]
    with open(categories_file, 'w') as f:
        f.write('\n'.join(categories))
        logger.info('Write {} categories to categories file at: {}'.format(len(categories), categories_file))


# Cell
def configure_logging(logging_level=logging.INFO):
    """
    Configures logging for the system.

    :param logging_level: The logging level to use.
    """
    logging.basicConfig(level=logging_level)


# Cell
if __name__ == '__main__' and '__file__' in globals():
    # for direct shell execution
    configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("categories",
                        help="The path to the categories file.")

    args = parser.parse_args()

    print(read_categories(args.categories))