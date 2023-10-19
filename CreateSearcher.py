from tflite_model_maker import searcher
import tensorflow as tf
from tflite_support.task import vision

data_loader = searcher.ImageDataLoader.create("/media/anlab/data-2tb/ANLAB_THUY/Serverless_Search/Model/1810_480x480_300x300_cirtorch_fl16_resnet101_WithoutNorm.tflite")
data_loader.load_from_folder("/media/anlab/data-2tb/ANLAB_THUY/Serverless_Search/data-images/data-images")
scann_options = searcher.ScaNNOptions(
      distance_measure="squared_l2",
      tree=searcher.Tree(num_leaves=44, num_leaves_to_search=4),
      score_ah=searcher.ScoreAH(dimensions_per_block=1, anisotropic_quantization_threshold=0.2))
#Change model extract feature
# data_loader._embedder_path = '1910_400x400_300x300_cirtorch_fl16_resnet101_WithoutNorm.tflite'

model = searcher.Searcher.create_from_data(data_loader, scann_options)
model.export(
      export_filename="1810_480x480_300x300_cirtorch_fl16_resnet101_WithoutNorm_ScanNN_SquareL2_100leaves.tflite",
      userinfo="",
      export_format=searcher.ExportFormat.TFLITE)
# Initialization
image_searcher = vision.ImageSearcher.create_from_file('1810_480x480_300x300_cirtorch_fl16_resnet101_WithoutNorm_ScanNN_SquareL2_100leaves.tflite')
# Run inference
image = vision.TensorImage.create_from_file('/home/anlab/Downloads/F093B51E-58D7-473F-8202-51044F4C7F0F.png')
result = image_searcher.search(image)
print(result.nearest_neighbors)