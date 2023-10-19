from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb

""" ... """
"""Creates the metadata for an image classifier."""

# Creates model info.
model_meta = _metadata_fb.ModelMetadataT()
model_meta.name = "MobileNetV1 image classifier"
model_meta.description = ("Identify the most prominent object in the "
                          "image from a set of 1,001 categories such as "
                          "trees, animals, food, vehicles, person etc.")
model_meta.version = "v1"
model_meta.author = "TensorFlow"
model_meta.license = ("Apache License. Version 2.0 "
                      "http://www.apache.org/licenses/LICENSE-2.0.")
# Creates input info.
input_meta = _metadata_fb.TensorMetadataT()

# Creates output info.
output_meta = _metadata_fb.TensorMetadataT()
input_meta.name = "image"
input_meta.description = (
    "Input image to be classified. The expected image is {0} x {1}, with "
    "three channels (red, blue, and green) per pixel. Each value in the "
    "tensor is a single byte between 0 and 255.".format(160, 160))
input_meta.content = _metadata_fb.ContentT()
input_meta.content.contentProperties = _metadata_fb.ImagePropertiesT()
input_meta.content.contentProperties.colorSpace = (
    _metadata_fb.ColorSpaceType.RGB)
input_meta.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.ImageProperties)
input_normalization = _metadata_fb.ProcessUnitT()
input_normalization.optionsType = (
    _metadata_fb.ProcessUnitOptions.NormalizationOptions)
input_normalization.options = _metadata_fb.NormalizationOptionsT()
input_normalization.options.mean = [0]
input_normalization.options.std = [255]
input_meta.processUnits = [input_normalization]
input_stats = _metadata_fb.StatsT()
input_stats.max = [1]
input_stats.min = [0]
input_meta.stats = input_stats
# Creates output info.
output_meta = _metadata_fb.TensorMetadataT()
output_meta.name = "feature"
output_meta.description = "Feature vector of the input image."
output_meta.content = _metadata_fb.ContentT()
output_meta.content.content_properties_type = _metadata_fb.FeatureProperties()
output_meta.content.contentPropertiesType = (
    _metadata_fb.ContentProperties.FeatureProperties)
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
populator = _metadata.MetadataPopulator.with_model_file('/media/anlab/data-2tb/ANLAB_THUY/Serverless_Search/Model/1810_480x480_300x300_cirtorch_fl16_resnet101_WithoutNorm.tflite')
populator.load_metadata_buffer(metadata_buf)
populator.populate()
import os
displayer = _metadata.MetadataDisplayer.with_model_file('/media/anlab/data-2tb/ANLAB_THUY/Serverless_Search/Model/1810_480x480_300x300_cirtorch_fl16_resnet101_WithoutNorm.tflite')
export_json_file = '/media/anlab/data-2tb/ANLAB_THUY/Serverless_Search/Model/1810_480x480_300x300_cirtorch_fl16_resnet101_WithoutNorm_metadata' + ".json"
json_file = displayer.get_metadata_json()
# Optional: write out the metadata as a json file
with open(export_json_file, "w") as f:
  f.write(json_file)