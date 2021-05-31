# AUTOGENERATED! DO NOT EDIT! File to edit: annotation_converter.ipynb (unless otherwise specified).

__all__ = ['convert', 'configure_logging']


# Cell
import sys
import argparse
import logging
from .core import list_subclasses, parse_known_args_with_help
from aiforce import annotation as annotation_package
from .annotation.core import AnnotationAdapter, SubsetType


# Cell
def convert(input_adapter: AnnotationAdapter, output_adapter: AnnotationAdapter):
    """
    Convert input annotations to output annotations.
    `input_adapter`: the input annotation adapter
    `output_adapter`: the output annotation adapter
    """
    categories = input_adapter.read_categories()
    annotations = input_adapter.read_annotations(SubsetType.TRAINVAL)
    output_adapter.write_categories(categories)
    output_adapter.write_annotations(annotations, SubsetType.TRAINVAL)


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
    parser.add_argument("-i",
                        "--input_adapter",
                        help="The annotation adapter to read the annotations.",
                        type=str,
                        choices=adapters.keys())
    parser.add_argument("-o",
                        "--output_adapter",
                        help="The annotation adapter to write the annotations.",
                        type=str,
                        choices=adapters.keys())

    argv = sys.argv
    args, argv = parse_known_args_with_help(parser, argv)
    input_adapter_class = adapters[args.input_adapter]
    output_adapter_class = adapters[args.output_adapter]

    # parse the input arguments
    input_parser = getattr(input_adapter_class, 'argparse')(prefix='input')
    input_args, argv = parse_known_args_with_help(input_parser, argv)

    # parse the output arguments
    output_parser = getattr(output_adapter_class, 'argparse')(prefix='output')
    output_args, argv = parse_known_args_with_help(output_parser, argv)

    convert(input_adapter_class(**vars(input_args)), output_adapter_class(**vars(output_args)))
