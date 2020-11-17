# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/tensorflow-tflite_metadata.ipynb (unless otherwise specified).

__all__ = ['logger', 'write_metadata', 'read_metadata', 'configure_logging']

# Cell

import logging
import logging.handlers
import argparse
import sys
import tensorflow as tf
from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb
from os.path import join, basename

# Cell

logger = logging.getLogger(__name__)

# Cell


def write_metadata(model_path, categories_file_path):
    """
    Write metadata to the Tensowflow Lite model on disk.
    `model_path`: the path to the Tensorflow Lite model
    `categories_file_path`: the path to the categories.txt file
    """
    model_meta = _metadata_fb.ModelMetadataT()

    # Creates input info.
    input_meta = _metadata_fb.TensorMetadataT()

    output_meta = _metadata_fb.TensorMetadataT()
    label_file = _metadata_fb.AssociatedFileT()
    label_file.name = basename(categories_file_path)
    label_file.description = "Labels for objects that the model can recognize."
    label_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS
    output_meta.associatedFiles = [categories_file_path]

    # Creates subgraph info.
    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [input_meta]
    subgraph.outputTensorMetadata = [output_meta]
    model_meta.subgraphMetadata = [subgraph]

    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_buf = b.Output()

    populator = tflite_metadata.MetadataPopulator.with_model_file(model_path)
    populator.load_metadata_buffer(metadata_buf)
    populator.load_associated_files([categories_file_path])
    populator.populate()

# Cell


def read_metadata(model_path):
    """
    Read meta-data from the Tensowflow Lite model on disk.
    `model_path`: the path to the Tensorflow Lite model
    returns: the metadata in JSON format
    """
    displayer = _metadata.MetadataDisplayer.with_model_file(model_path)
    metadata = displayer.get_metadata_json()
    return metadata

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
    parser.add_argument("model",
                        help="The path to the Tensorflow Lite exported model file.")
    parser.add_argument("-c",
                        "--categories",
                        help="The categories file to add to the Tensorflow Lite model.",
                        type=str,
                        default=None)
    args = parser.parse_args()

    if args.categories is None:
        metadata = read_metadata(args.model)
        logger.info('Read metadata from Tensorflow Lite model: {}'.format(args.model))
        logger.info(metadata)
    else:
        write_metadata(args.model, args.categories)

    logger.info('FINISHED!!!')
