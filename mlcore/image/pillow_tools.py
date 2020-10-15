# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/image-pillow_tools.ipynb (unless otherwise specified).

__all__ = ['EXIF_ORIENTATION_TAG', 'logger', 'limit_to_max_size', 'fit_to_max_size', 'get_image_size',
           'get_image_orientation', 'read_exif_metadata', 'write_exif_metadata', 'assign_exif_orientation',
           'convert_to_base64', 'configure_logging']

# Cell
import sys
import argparse
import logging
import base64
import piexif
from io import BytesIO
from PIL import Image as PILImage
from .tools import ImageOrientation
from ..io.core import scan_files

# Cell

EXIF_ORIENTATION_TAG = 'Orientation'
"""The Image EXIF orientation tag"""

# Cell

logger = logging.getLogger(__name__)

# Cell


def limit_to_max_size(img, max_size):
    """
    Limit the image size to max size and scale the image,
    if max size exceeded.
    `img`: The image to validate as Pillow Image.
    `max_size`: The max allowed image size.
    :return: The eventually resized image.
    """
    biggest_size = max(img.size)
    if max_size and biggest_size > max_size:
        ratio = 1.0 * max_size / biggest_size
        img = img.resize([int(ratio * s) for s in img.size])
    return img

# Cell


def fit_to_max_size(img, max_width, max_height):
    """
    Limit the image size to maximum width and height and scale the image,
    if size exceeded.
    `img`: The image to validate as Pillow Image.
    `max_width`: The max allowed image width.
    `max_height`: The max allowed image height.
    :return: The eventually resized image.
    """
    w, h = img.size
    scale_delta = max(w - max_width, h - max_height)
    if scale_delta > 0:
        max_size = max(w - scale_delta, h - scale_delta)
        img = limit_to_max_size(img, max_size)
    return img

# Cell


def get_image_size(fname):
    """
    Calculates image size of a given image file path.
    `fname`: the file path
    return: the Pillow image, image width and height
    """
    image = PILImage.open(fname)
    w, h = image.size
    return image, w, h

# Cell


def get_image_orientation(fname):
    """
    Parses the EXIF orientation information from the image.
    `fname`: the file path
    :return: The Pillow image and the orientation of the image.
    """
    orientation = ImageOrientation.TOP
    image, exif_data = read_exif_metadata(fname)
    if exif_data is not None and "0th" in exif_data:
        exif_data_0 = exif_data["0th"]
        if piexif.ImageIFD.Orientation in exif_data_0:
            try:
                orientation = ImageOrientation(exif_data_0[piexif.ImageIFD.Orientation])
            except ValueError as e:
                logger.error(e)
    return image, orientation

# Cell


def read_exif_metadata(fname):
    """
    Read the EXIF metadata information from the image.
    `fname`: the file path
    :return: The Pillow image, EXIF metadata as dictionary or None, if no EXIF data exist.
    """
    image = PILImage.open(fname)
    exif_data = None
    if "exif" in image.info:
        exif_data = piexif.load(image.info["exif"])
    return image, exif_data

# Cell


def write_exif_metadata(image, exif_data, fname):
    """
    Write the EXIF metadata information to the image.
    `image`: the Pillow image to write the EXIF metadata to
    `exif_data`: the EXIF metadata as dictionary
    `fname`: a file path to store the image
    :return: `True` if EXIF metadata saved, else `False`
    """
    if image:
        piexif.dump(exif_data)
        exif_bytes = piexif.dump(exif_data)
        image.save(fname, exif=exif_bytes)
        return True
    return False

# Cell


def assign_exif_orientation(fname):
    """
    Parses the EXIF orientation metadata from the image,
    rotate the image accordingly and remove the image EXIF orientation metadata.
    `fname`: the file path
    :return: `True` if EXIF metadata saved, else `False`.
    """
    image, exif_data = read_exif_metadata(fname)

    if exif_data and piexif.ImageIFD.Orientation in exif_data["0th"]:
        orientation = exif_data["0th"].pop(piexif.ImageIFD.Orientation)
        try:
            orientation = ImageOrientation(orientation)
        except ValueError as e:
            logger.error(e)
            orientation = ImageOrientation.TOP

        if orientation == ImageOrientation.TOP_FLIPPED:
            image = image.transpose(PILImage.FLIP_LEFT_RIGHT)
        elif orientation == ImageOrientation.BOTTOM:
            image = image.rotate(180)
        elif orientation == ImageOrientation.BOTTOM_FLIPPED:
            image = image.rotate(180).transpose(PILImage.FLIP_LEFT_RIGHT)
        elif orientation == ImageOrientation.RIGHT_FLIPPED:
            image = image.rotate(-90, expand=True).transpose(PILImage.FLIP_LEFT_RIGHT)
        elif orientation == ImageOrientation.RIGHT:
            image = image.rotate(-90, expand=True)
        elif orientation == ImageOrientation.LEFT_FLIPPED:
            image = image.rotate(90, expand=True).transpose(PILImage.FLIP_LEFT_RIGHT)
        elif orientation == ImageOrientation.LEFT:
            image = image.rotate(90, expand=True)

    return write_exif_metadata(image, exif_data, fname)

# Cell


def convert_to_base64(image, image_type="PNG"):
    """
    Converts the specified image into a base64 version of itself.

    `image`: The image to transform as Pillow Image.
    `image_type`: The image type.
    :return: The base64 encoded version of the image.
    """
    buffered = BytesIO()
    image.save(buffered, format=image_type)
    return base64.b64encode(buffered.getvalue()).decode('UTF-8')

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
    # for direct shell execution
    configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path",
                        help="The path to the image files.")

    args = parser.parse_args()
    files = scan_files(args.image_path)
    for file in files:
        _, w, h = get_image_size(file)
        _, orientation = get_image_orientation(file)
        logger.info("Size: width: {}, height: {}, orientation: {}".format(w, h, orientation))
