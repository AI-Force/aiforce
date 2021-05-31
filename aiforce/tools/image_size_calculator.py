# AUTOGENERATED! DO NOT EDIT! File to edit: tools-image_size_calculator.ipynb (unless otherwise specified).

__all__ = ['logger', 'ImageSizeCalculator', 'configure_logging']


# Cell
import logging
import logging.handlers
import argparse
import sys
from functools import reduce
from ..image.pillow_tools import get_image_size
from ..io.core import scan_path


# Cell
logger = logging.getLogger(__name__)


# Cell
class ImageSizeCalculator:
    """
    Calculates image sizes in a given folder and subfolders. Summarize unique image sizes at the end.
    `path`: the folder to process
    """

    def __init__(self, path):
        self.path = path

    def calculate(self):
        """
        The main logic.
        return: a dictionary with unique sizes as key and image count as value
        """

        images = scan_path(self.path)
        all_images = len(images)
        unique_sizes = {}

        for index, image in enumerate(images):
            _, width, height = get_image_size(image)
            size_key = "{}x{}".format(width, height)
            logger.info("{} / {} - Handle Image {} with size {}x{}".format(
                index + 1,
                all_images,
                image,
                width,
                height,
            ))
            if size_key not in unique_sizes:
                unique_sizes[size_key] = 0
            unique_sizes[size_key] += 1

        return unique_sizes


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


# Cell
if __name__ == '__main__' and '__file__' in globals():
    configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="The folder to scan.")
    args = parser.parse_args()

    calculator = ImageSizeCalculator(args.folder)
    sizes = calculator.calculate()
    image_count = reduce(lambda a, x: a + x, sizes.values())
    logger.info("Images Analyzed: {}".format(image_count))

    logger.info("Unique Image Sizes and Image Count:")
    for key, size in sizes.items():
        logger.info("{}: {}".format(key, size))

    logger.info('FINISHED!!!')
