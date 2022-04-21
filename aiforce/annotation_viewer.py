# AUTOGENERATED! DO NOT EDIT! File to edit: annotation_viewer.ipynb (unless otherwise specified).

__all__ = ['WINDOW_NAME', 'ANNOTATION_COLOR', 'ImageLoader', 'show_annotated_images', 'configure_logging']


# Cell
import sys
import argparse
import logging
import cv2
import math
import numpy as np
import aiforce.image.opencv_tools as opencv_tools
import aiforce.image.pillow_tools as pillow_tools
from enum import Enum
from os.path import basename
from aiforce import annotation as annotation_package
from .core import list_subclasses, parse_known_args_with_help
from .annotation.core import AnnotationAdapter, annotation_filter, SubsetType, RegionShape


# Cell

# the name of the opencv window
WINDOW_NAME = 'Annotation'
# the color of the annotations
ANNOTATION_COLOR = (0, 255, 255)


# Cell
class ImageLoader(Enum):
    """
    Currently supported image loader libraries.
    """
    OPEN_CV = 'open_cv'
    PILLOW = 'pillow'

    def __str__(self):
        return self.value


# Cell
def show_annotated_images(annotation_adapter, subset_type, image_loader, max_width=0, max_height=0, filter_names=None):
    """
    Show images with corresponding annotations.
    Images are shown one at a time with switching by using the arrow left/right keys.
    `annotation_adapter`: The annotation adapter to use
    `subset_type`: The subset to load
    `image_loader`: The image loader library to use
    `max_width`: The maximum width to scale the image for visibility.
    `max_height`: The maximum height to scale the image for visibility.
    """
    categories = annotation_adapter.read_categories()
    annotations = annotation_adapter.read_annotations(subset_type)

    if filter_names:
        annotations = annotation_filter(annotations, lambda _, anno: basename(anno.file_path) in filter_names)

    len_annotations = len(annotations)

    if len_annotations == 0:
        logging.error("No Annotations found")
        return

    logging.info("Load images with {}".format(image_loader))

    i = 0
    annotation_keys = list(annotations.keys())

    logging.info("Keys to use:")
    logging.info("n = Next Image")
    logging.info("b = Previous Image")
    logging.info("q = Quit")

    logging.info("Annotations to view: {}".format(len_annotations))

    while True:
        annotation_id = annotation_keys[i]
        annotation = annotations[annotation_id]
        logging.info("View Image {}/{}: {}".format(i + 1, len_annotations, annotation.file_path))
        if image_loader == ImageLoader.PILLOW:
            img, width, height = pillow_tools.get_image_size(annotation.file_path)
            img = opencv_tools.from_pillow_image(img)
        elif image_loader == ImageLoader.OPEN_CV:
            img, width, height = opencv_tools.get_image_size(annotation.file_path)
        else:
            logging.error("Unsupported image loader")
            img = None
            width = 0
            height = 0

        if img is None:
            logging.info("Image not found at {}".format(annotation.file_path))
            img = np.zeros(shape=(1, 1, 3))
        else:
            logging.info("Image size (WIDTH x HEIGHT): ({} x {})".format(width, height))

        if annotation.regions:
            logging.info("Found {} regions".format(len(annotation.regions)))
            for region_index, region in enumerate(annotation.regions):
                points = list(zip(region.points_x, region.points_y))
                logging.info("Found {} of category {} with {} points: {}".format(region.shape,
                                                                                 ','.join(region.labels),
                                                                                 len(points), points))
                if region.shape == RegionShape.CIRCLE:
                    img = cv2.circle(img, points[0], int(region.radius_x), ANNOTATION_COLOR, 2)
                elif region.shape == RegionShape.ELLIPSE:
                    angle = region.rotation * 180 // math.pi
                    img = cv2.ellipse(img, points[0], (int(region.radius_x), int(region.radius_y)), angle, 0, 360,
                                      ANNOTATION_COLOR, 2)
                elif region.shape == RegionShape.POINT:
                    img = cv2.circle(img, points[0], 1, ANNOTATION_COLOR, 2)
                elif region.shape == RegionShape.POLYGON or region.shape == RegionShape.POLYLINE:
                    pts = np.array(points, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    img = cv2.polylines(img, [pts], region.shape == RegionShape.POLYGON, ANNOTATION_COLOR, 2)
                elif region.shape == RegionShape.RECTANGLE:
                    img = cv2.rectangle(img, points[0], points[1], ANNOTATION_COLOR, 2)

        if max_width and max_height:
            img = opencv_tools.fit_to_max_size(img, max_width, max_height)

        cv2.imshow(WINDOW_NAME, img)
        cv2.setWindowTitle(WINDOW_NAME, "Image {}/{}".format(i + 1, len_annotations))

        k = cv2.waitKey(0)
        if k == ord('q'):    # 'q' key to stop
            break
        elif k == ord('b'):
            i = max(0, i - 1)
        elif k == ord('n'):
            i = min(len_annotations - 1, i + 1)

    cv2.destroyAllWindows()


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

    # read annotation adapters to use
    adapters = list_subclasses(annotation_package, AnnotationAdapter)

    parser = argparse.ArgumentParser()
    parser.add_argument("-a",
                        "--annotation",
                        help="The annotation adapter to read the annotations.",
                        type=str,
                        choices=adapters.keys(),
                        required=True)
    parser.add_argument("--image_loader",
                        help="The image library for reading the image.",
                        choices=list(ImageLoader),
                        type=ImageLoader,
                        default=ImageLoader.PILLOW)
    parser.add_argument("--subset",
                        help="The image subset to read.",
                        choices=list(SubsetType),
                        type=SubsetType,
                        default=SubsetType.TRAINVAL)
    parser.add_argument("--max-width",
                        help="The maximum width to scale the image for visibility.",
                        type=int,
                        default=0)
    parser.add_argument("--max-height",
                        help="The maximum height to scale the image for visibility.",
                        type=int,
                        default=0)
    parser.add_argument("--filter",
                        help="Filter file names to view.",
                        nargs="*",
                        default=[])
    argv = sys.argv
    args, argv = parse_known_args_with_help(parser, argv)

    adapter_class = adapters[args.annotation]

    # parse the annotation arguments
    annotation_parser = getattr(adapter_class, 'argparse')()
    annotation_args, argv = parse_known_args_with_help(annotation_parser, argv)

    show_annotated_images(adapter_class(**vars(annotation_args)), args.subset, args.image_loader, args.max_width,
                          args.max_height, args.filter)