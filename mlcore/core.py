# AUTOGENERATED! DO NOT EDIT! File to edit: core.ipynb (unless otherwise specified).

__all__ = ['Type']

# Cell

from enum import Enum

# Cell


class Type(Enum):
    """
    Currently supported Machine Learning Types.
    """
    IMAGE_CLASSIFICATION = 'image_classification'
    IMAGE_SEGMENTATION = 'image_segmentation'
    IMAGE_GENERATION = 'image_generation'

    def __str__(self):
        return self.value
