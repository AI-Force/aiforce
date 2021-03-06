# AUTOGENERATED! DO NOT EDIT! File to edit: image-tools.ipynb (unless otherwise specified).

__all__ = ['ImageOrientation', 'get_image_scale']


# Cell
from enum import Enum


# Cell
class ImageOrientation(Enum):
    """
    Image EXIF orientations.
    """
    TOP = 1
    TOP_FLIPPED = 2
    BOTTOM = 3
    BOTTOM_FLIPPED = 4
    RIGHT_FLIPPED = 5
    RIGHT = 6
    LEFT_FLIPPED = 7
    LEFT = 8

    def __str__(self):
        return str(self.value)


# Cell
def get_image_scale(image_size, target_size):
    """
    Calculates the scale of the image to fit the target size.
    `image_size`: The image size as tuple of (w, h)
    `target_size`: The target size as tuple of (w, h).
    :return: The image scale as tuple of (w_scale, h_scale)
    """
    (image_w, image_h) = image_size
    (target_w, target_h) = target_size
    scale = (target_w / float(image_w), target_h / float(image_h))
    return scale
