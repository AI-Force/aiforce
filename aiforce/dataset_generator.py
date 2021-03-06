# AUTOGENERATED! DO NOT EDIT! File to edit: dataset_generator.ipynb (unless otherwise specified).

__all__ = ['logger', 'configure_logging', 'generate']


# Cell
import sys
import argparse
import logging
from os.path import join
from datetime import datetime
from logging.handlers import MemoryHandler
from .core import list_subclasses, parse_known_args_with_help
from aiforce import annotation as annotation_package
from .annotation.core import AnnotationAdapter
from aiforce import dataset as dataset_package
from .dataset.core import Dataset


# Cell
logger = logging.getLogger('aiforce.dataset')


# Cell
def configure_logging(logging_level=logging.INFO):
    """
    Configures logging for the system.
    """
    logging.basicConfig(level=logging_level)

    log_memory_handler = MemoryHandler(1, flushLevel=logging_level)
    log_memory_handler.setLevel(logging_level)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)

    logger.addHandler(log_memory_handler)
    logger.addHandler(stdout_handler)

    logger.setLevel(logging_level)

    return log_memory_handler


# Cell
def generate(dataset: Dataset, log_memory_handler):
    """
    Generate a dataset.
    `dataset`: the dataset to build
    `log_memory_handler`: the log handler for the build log
    """
    dataset.build_info()

    logger.info('Start build {} at {}'.format(type(dataset).__name__, dataset.output_adapter.path))

    dataset.create_folders()

    # create the build log file
    log_file_name = datetime.now().strftime("build_%Y.%m.%d-%H.%M.%S.log")
    file_handler = logging.FileHandler(join(dataset.output_adapter.path, log_file_name), encoding="utf-8")
    log_memory_handler.setTarget(file_handler)

    dataset.build()

    logger.info('Finished build {} at {}'.format(type(dataset).__name__, dataset.output_adapter.path))


# Cell
if __name__ == '__main__' and '__file__' in globals():
    # for direct shell execution
    log_handler = configure_logging()

    # read annotation adapters to use
    adapters = list_subclasses(annotation_package, AnnotationAdapter)

    # read datasets to use
    datasets = list_subclasses(dataset_package, Dataset)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--input",
                        help="The annotation input adapter.",
                        type=str,
                        choices=adapters.keys(),
                        required=True)
    parser.add_argument("-d",
                        "--dataset",
                        help="The dataset to generate.",
                        type=str,
                        choices=datasets.keys(),
                        required=True)
    parser.add_argument("-o",
                        "--output",
                        help="The annotation output adapter.",
                        type=str,
                        choices=adapters.keys(),
                        required=True)

    argv = sys.argv
    args, argv = parse_known_args_with_help(parser, argv)
    input_adapter_class = adapters[args.input]
    dataset_class = datasets[args.dataset]
    output_adapter_class = adapters[args.output]

    # parse the input arguments
    input_parser = getattr(input_adapter_class, 'argparse')(prefix='input')
    input_args, argv = parse_known_args_with_help(input_parser, argv)

    # parse the dataset arguments
    dataset_parser = getattr(dataset_class, 'argparse')()
    dataset_args, argv = parse_known_args_with_help(dataset_parser, argv)

    # parse the output arguments
    output_parser = getattr(output_adapter_class, 'argparse')(prefix='output')
    output_args, argv = parse_known_args_with_help(output_parser, argv)

    input_adapter = input_adapter_class(**vars(input_args))
    output_adapter = output_adapter_class(**vars(output_args))
    dataset_args.input_adapter = input_adapter
    dataset_args.output_adapter = output_adapter
    target_dataset = dataset_class(**vars(dataset_args))

    logger.info('Build parameters:')
    logger.info(' '.join(sys.argv[1:]))

    generate(target_dataset, log_handler)