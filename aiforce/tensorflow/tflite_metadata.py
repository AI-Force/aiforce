# AUTOGENERATED! DO NOT EDIT! File to edit: tensorflow-tflite_metadata.ipynb (unless otherwise specified).

__all__ = ['SAVED_MODEL_META_DEFAULT_KEY', 'AUTHOR', 'logger', 'MetaInfo', 'create_metadata', 'write_metadata',
           'read_metadata', 'configure_logging']


# Cell
import logging
import logging.handlers
import argparse
import sys
import tensorflow as tf
from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb
from os.path import basename
from mlcore.dataset.type import DatasetType, infer_dataset_type


# Cell
SAVED_MODEL_META_DEFAULT_KEY = 'serving_default'
AUTHOR = 'Protosolution'


# Cell
logger = logging.getLogger(__name__)


# Cell
class MetaInfo:
    """
    Metadata information.
    `name`: The metadata name.
    `desc`: The metadata description.
    `prop`: The metadata property.
    `prop_type`: The metadata property type.
    `range_min`: The metadata min range.
    `range_max`: The metadata max range.
    `stats_min`: A list of min statistics per channel.
    `stats_max`: A list of max statistics per channel.
    `associated_files`: A list of associated files.
    """

    def __init__(self, name=None, desc=None, prop=None, prop_type=None, range_min=None, range_max=None,
                 stats_min=None, stats_max=None, associated_files=None):
        self.name = name
        self.desc = desc
        self.prop = prop
        self.prop_type = prop_type
        self.range_min = range_min
        self.range_max = range_max
        self.stats_min = stats_min
        self.stats_max = stats_max
        self.associated_files = associated_files

    def has_range(self):
        return self.range_min is not None or self.range_max is not None

    def has_stats(self):
        return self.stats_min or self.stats_max


# Cell
def create_metadata(saved_model_dir, categories_file_path, model_type, model_name, model_version=1):
    """
    Write metadata to the Tensowflow Lite model on disk.
    `saved_model_dir`: the path to the folder containing the SavedModel
    `categories_file_path`: the path to the categories.txt file
    `model_type`: the type of the model
    `model_name`: the name of the model
    `model_version`: the version of the model
    returns: the model metadata
    """
    saved_model = tf.saved_model.load(saved_model_dir)
    saved_model_meta = saved_model.signatures[SAVED_MODEL_META_DEFAULT_KEY]

    # Creates model info.
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = model_name

    description = ""
    if model_type == DatasetType.IMAGE_OBJECT_DETECTION:
        description = ("Identify which of a known set of objects "
                       "might be present and provide information about their positions "
                       "within the given image or a video stream.")
    elif model_type == DatasetType.IMAGE_CLASSIFICATION:
        description = ("Identify the most prominent object in the "
                       "image from a set of ategories.")

    model_meta.description = description
    model_meta.version = "v{}".format(model_version)
    model_meta.author = AUTHOR
    model_meta.license = ("Apache License. Version 2.0 "
                          "http://www.apache.org/licenses/LICENSE-2.0.")

    # Creates input info.
    input_meta = _create_input_metadata(saved_model_meta)

    # Creates output info.
    output_meta, output_groups = _create_output_metadata(saved_model_meta, categories_file_path, model_type)

    # Creates subgraph info.
    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = input_meta
    subgraph.outputTensorMetadata = output_meta
    subgraph.outputTensorGroups = output_groups
    model_meta.subgraphMetadata = [subgraph]

    return model_meta


# Cell
def write_metadata(model_meta, model_path, categories_file_path):
    """
    Write metadata to the Tensowflow Lite model on disk.
    `model_meta`: the model metadata
    `model_path`: the path to the Tensorflow Lite model
    `categories_file_path`: the path to the categories.txt file
    """

    b = flatbuffers.Builder(0)
    b.Finish(
        model_meta.Pack(b),
        _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_buf = b.Output()

    populator = _metadata.MetadataPopulator.with_model_file(model_path)
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
def _create_input_metadata(saved_model_meta, input_min=0, input_max=255, norm_mean=127.5, norm_std=127.5):
    """
    Creates input metadata

    `saved_model_meta`: The saved model meta data.
    `input_min`: The input min value.
    `input_max`: The input max value.
    `norm_mean`: The normalization mean value.
    `norm_std`: The normalization std value.
    """
    model_input = saved_model_meta.inputs[0]
    _, width, height, channel = model_input.shape

    if channel == 1:
        color_space = _metadata_fb.ColorSpaceType.GRAYSCALE
        channel_description = "one channel (grayscale)"
    if channel == 3:
        color_space = _metadata_fb.ColorSpaceType.RGB
        channel_description = "three channels (red, blue, and green)"
    else:
        color_space = _metadata_fb.ColorSpaceType.UNKNOWN
        channel_description = "{} channel".format(channel)

    # Creates input info.
    desc = (
        "Input image. The expected image is {0} x {1}, with "
        "{2} per pixel. Each value in the tensor "
        "is between {3} and {4}.".format(width, height, channel_description, input_min, input_max))
    image_prop = _metadata_fb.ImagePropertiesT()
    image_prop.colorSpace = color_space

    input_normalization = _metadata_fb.ProcessUnitT()
    input_normalization.optionsType = (
        _metadata_fb.ProcessUnitOptions.NormalizationOptions)
    input_normalization.options = _metadata_fb.NormalizationOptionsT()
    input_normalization.options.mean = [norm_mean]
    input_normalization.options.std = [norm_std]

    meta_info = MetaInfo(name="image",
                         desc=desc,
                         prop=image_prop,
                         prop_type=_metadata_fb.ContentProperties.ImageProperties,
                         stats_min=[input_min],
                         stats_max=[input_max])

    input_meta = _create_tensor_metadata(meta_info)
    input_meta.processUnits = [input_normalization]

    return [input_meta]


# Cell
def _create_output_metadata(saved_model_meta, categories_file_path, model_type):
    """
    Creates output metadata

    `saved_model_meta`: The saved model meta data.
    `categories_file_path`: the path to the categories.txt file.
    `model_type`: The type of the model.
    """
    output_meta = []
    output_groups = []

    if model_type == Type.IMAGE_OBJECT_DETECTION:
        output_meta = [
            MetaInfo(name="location",
                     desc="The locations of the detected boxes.",
                     prop=_create_bbox_content_property_metadata(),
                     prop_type=_metadata_fb.ContentProperties.BoundingBoxProperties,
                     range_min=2,
                     range_max=2),
            MetaInfo(name="category",
                     desc="The categories of the detected boxes.",
                     prop=_metadata_fb.FeaturePropertiesT(),
                     prop_type=_metadata_fb.ContentProperties.FeatureProperties,
                     range_min=2,
                     range_max=2,
                     associated_files=[
                         _create_associated_files_metadata(categories_file_path,
                                                           "Labels for objects that the model can recognize.",
                                                           _metadata_fb.AssociatedFileType.TENSOR_VALUE_LABELS)
                     ]),
            MetaInfo(name="score",
                     desc="The scores of the detected boxes.",
                     prop=_metadata_fb.FeaturePropertiesT(),
                     prop_type=_metadata_fb.ContentProperties.FeatureProperties,
                     range_min=2,
                     range_max=2),
            MetaInfo(name="number of detections",
                     desc="The number of the detected boxes.",
                     prop=_metadata_fb.FeaturePropertiesT(),
                     prop_type=_metadata_fb.ContentProperties.FeatureProperties),
        ]
        output_groups = _metadata_fb.TensorGroupT()
        output_groups.name = "detection result"
        output_groups.tensorNames = [o.name for o in output_meta[:3]]
        output_groups = [output_groups]

    elif model_type == Type.IMAGE_CLASSIFICATION:
        output_meta = [
            MetaInfo(name="probability",
                     desc="Probabilities of the labels respectively.",
                     prop=_metadata_fb.FeaturePropertiesT(),
                     prop_type=_metadata_fb.ContentProperties.FeatureProperties,
                     stats_min=[0.0],
                     stats_max=[1.0],
                     associated_files=[
                         _create_associated_files_metadata(categories_file_path,
                                                           "Labels for objects that the model can recognize.")
                     ]),
        ]

    if output_meta:
        output_meta = [_create_tensor_metadata(m) for m in output_meta]

    return output_meta, output_groups


# Cell
def _create_tensor_metadata(meta_info: MetaInfo):
    """
    Creates tensor metadata

    `meta_info`: The metadata information to create a tensor metadata for
    returns: The tensor metadata.
    """
    meta = _metadata_fb.TensorMetadataT()
    meta.name = meta_info.name
    meta.description = meta_info.desc

    meta.content = _metadata_fb.ContentT()
    meta.content.content_properties = meta_info.prop
    meta.content.contentPropertiesType = meta_info.prop_type
    meta.associatedFiles = meta_info.associated_files
    if meta_info.has_range:
        meta.content.range = _metadata_fb.ValueRangeT()
        meta.content.range.min = 0 if meta_info.range_min is None else meta_info.range_min
        meta.content.range.max = 0 if meta_info.range_max is None else meta_info.range_max

    if meta_info.has_stats:
        meta.stats = _metadata_fb.StatsT()
        meta.stats.max = meta_info.stats_max
        meta.stats.min = meta_info.stats_min

    return meta


# Cell
def _create_bbox_content_property_metadata(bbox_type=None, bbox_index=None):
    """
    Creates bounding box property content metadata
    `bbox_type`: The bounding box type.
    `bbox_index`: The bounding box index.
    returns: The bounding box property content.
    """
    properties = _metadata_fb.BoundingBoxPropertiesT()
    properties.index = [1, 0, 3, 2] if bbox_index is None else bbox_index
    properties.type = _metadata_fb.BoundingBoxType.BOUNDARIES if bbox_type is None else bbox_type
    return properties


# Cell
def _create_associated_files_metadata(categories_file_path, desc, label_type=None):
    """
    Creates associated files metadata
    `categories_file_path`: the path to the categories.txt file
    `desc`: The tensor metadata description.
    `label_type`: The label type.
    returns: The associated files metadata.
    """
    associated_file = _metadata_fb.AssociatedFileT()
    associated_file.name = basename(categories_file_path)
    associated_file.description = desc
    associated_file.type = _metadata_fb.AssociatedFileType.TENSOR_AXIS_LABELS if label_type is None else label_type
    return associated_file


# Cell
def configure_logging(logging_level=logging.INFO):
    """
    Configures logging for the system.

    `logging_level`: The logging level to use.
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
    parser.add_argument("-s",
                        "--source",
                        help="The path to the folder containing the SavedModel.",
                        type=str,
                        default=None)
    parser.add_argument("-c",
                        "--categories",
                        help="The categories file to add to the Tensorflow Lite model.",
                        type=str,
                        default=None)
    parser.add_argument("-n",
                        "--name",
                        help="The name of the model.",
                        type=str,
                        default=None)
    parser.add_argument("-v",
                        "--version",
                        help="The version of the model.",
                        type=int,
                        default=1)
    parser.add_argument("-t",
                        "--type",
                        help="The type of the model, if not explicitly set try to infer from categories file path.",
                        choices=list(DatasetType),
                        type=DatasetType,
                        default=None)

    args = parser.parse_args()

    if args.categories is None:
        metadata = read_metadata(args.model)
        logger.info('Read metadata from Tensorflow Lite model: {}'.format(args.model))
        logger.info(metadata)
    else:
        model_type = args.type

        # try to infer the model type if not explicitly set
        if model_type is None:
            try:
                model_type = infer_dataset_type(args.categories)
            except ValueError as e:
                logger.error(e)
                sys.exit(1)

        model_meta = create_metadata(args.source, args.categories, model_type, args.name, args.version)
        write_metadata(model_meta, args.model, args.categories)

    logger.info('FINISHED!!!')