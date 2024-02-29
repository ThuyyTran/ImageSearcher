import argparse
import fnmatch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import sys
import base64
import scipy
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
import coremltools
import coremltools as ct
import time
from coremltools.models.neural_network import quantization_utils
# import onnxruntime as ort
# import onnx
import torch
# from onnx_tf.backend import prepare
# import tensorflow as tf 
from torchvision.transforms import functional as F
# from torch.utils.model_zoo import load_url
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms
import cv2
sys.path.append('/media/anlab/data-2tb/ANLAB_THUY/ImageSearcher/SOLAR/')
from solar_global.networks.imageretrievalnet import init_network, extract_vectors
from solar_global.datasets.testdataset import configdataset
from solar_global.utils.download import download_test
from solar_global.utils.evaluate import compute_map_and_print
from solar_global.utils.general import get_data_root, htime
from solar_global.utils.networks import load_network
from solar_global.utils.plots import plot_ranks, plot_embeddings
from torchvision import transforms
import time
# from onnx_coreml import convert
import coremltools
import coremltools as ct

# from cirtorch.datasets.datahelpers import im_resize
import torch.nn as nn
# class Resize_ratio():
#     def __init__(self, imsize):
#         self.imsize = imsize
#     def __call__(self, image):
#         image = im_resize(image, self.imsize)
#         return image
import math
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(list1, list2):
    return dot(list1, list2) / (norm(list1) * norm(list2))

def cosine_distance(list1, list2):
    return 1 - cosine_similarity(list1, list2)

def euclidean_distance(list1, list2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(list1, list2)]))
class WDViTModel():
    def __init__(self, model_path='/media/anlab/data-2tb/ANLAB_THUY/lashinbang-server-dzungdk/ConvertModel/weights') -> None:
        
        self.model_norm, self.model_depth = self.load_model(model_path)
        _, height, width, _ = self.model_norm.inputs[0].shape
        self.height = height
        self.width = width
        
    def load_model(self, model_path):
        full_model = tf.keras.models.load_model(model_path)
        
        model_norm = tf.keras.models.Model(
           full_model.inputs,  full_model.get_layer("predictions_norm").output
        )
        
        model_depth = tf.keras.models.Model(
            # full_model.get_layer('predictions_norm').output, full_model.get_layer("predictions_stochdepth").output
            full_model.get_layer('predictions_norm').output, full_model.get_layer("predictions_stochdepth").output
        )
        
        return model_norm, model_depth
     
class Network(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    def forward(self,x):
        # x1 = F.resize(x, (300, 300))
        out1 = self.model(x)   
        reshaped_tensor1 = out1.view(1, 2048)
        # x2 = F.resize(x, (480, 480))
        # out2 = self.model(x2)   
        # reshaped_tensor2 = out2.view(1, 2048)
        # return (reshaped_tensor1+reshaped_tensor2)/2
        return reshaped_tensor1
#Sua kien truc model "/home/anlab/anaconda3/envs/testconvertmodel/lib/python3.7/site-packages/coremltools/models/neural_network/builder.py" 
# state = torch.load(os.path.join(get_data_root(), 'networks/model_best.pth.tar'),map_location=torch.device('cuda'))
# net_params = {}
# net_params['architecture'] = state['meta']['architecture']
# net_params['pooling'] = state['meta']['pooling'] 
# net_params['local_whitening'] = state['meta'].get('local_whitening', False)
# net_params['regional'] = state['meta'].get('regional', False)
# net_params['whitening'] = state['meta'].get('whitening', True)
# net_params['mean'] = state['meta']['mean']
# net_params['std'] = state['meta']['std']
# net_params['pretrained'] = False
# net = load_network('model_best.pth.tar')
# net.load_state_dict(state['state_dict'])
# # net.cuda()  
# net.eval()

# test_model = Network(net)
# test_model.eval()
# scale = 1/(0.226*255.0)
# bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]
# input_shape = ct.Shape(shape=(1, 3, 300, 300))
# dummy_input = torch.rand(1,3,300,300)
# input_tensor = ct.ImageType(name="my_input", shape=input_shape,scale=scale, bias=bias)
# traced_model = torch.jit.trace(test_model.eval(), dummy_input)
# traced_model.eval()
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:    
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    tf.config.set_visible_devices([], 'GPU')
# model = WDViTModel()
model_path='/media/anlab/data-2tb/ANLAB_THUY/lashinbang-server-dzungdk/ConvertModel/weights'
full_model = tf.keras.models.load_model(model_path)
model_norm = tf.keras.models.Model(
    full_model.inputs,  full_model.get_layer("predictions_norm").output
)
# model_depth = tf.keras.models.Model(
#     full_model.get_layer('predictions_norm').output, full_model.get_layer("predictions_stochdepth").output
# )
# combined_input = full_model.inputs
# combined_output = model_depth(model_norm(combined_input))
# combined_model = tf.keras.models.Model(inputs=combined_input, outputs=combined_output)
input_shape = ct.Shape(shape=(1, 448, 448,3))
input_tensor = ct.ImageType(name="input_1", shape=input_shape)
mlprogram = ct.convert(
    model_norm,
    # minimum_deployment_target=ct.target.iOS13,
    inputs=[input_tensor],
    # outputs=[ct.TensorType(name="embeddings")],
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.ALL,
    # pass_pipeline=ct.PassPipeline.DEFAULT_PRUNING
)

# from coremltools.models.neural_network import quantization_utils
# model_8bit = quantization_utils.quantize_weights(mlprogram, nbits=8)
# spec = model_8bit.get_spec()
# outputmodel = ct.models.MLModel(spec, weights_dir=model_8bit.weights_dir)
# saved_model = 'ModelConvert/TestModel/Tensorflow_ALL_fp8.mlpackage'
# outputmodel.save(saved_model)
compressed_model = ct.compression_utils.affine_quantize_weights(mlprogram)
saved_model = 'ModelConvert/TestModel/Tensorflow_cpu.mlpackage'
compressed_model.save(saved_model)
# import coremltools as ct
# import coremltools.optimize.coreml as cto
# # load model
# mlmodel = ct.models.MLModel(outputmodel)

# # define op config 
# op_config = cto.OpPalettizerConfig(mode="kmeans", nbits=6)

# # define optimization config by applying the op config globally to all ops 
# config = cto.OptimizationConfig(global_config=op_config)

# # palettize weights
# compressed_mlmodel = cto.palettize_weights(mlmodel, config)
# # from coremltools.optimize.coreml import (
# #     OpMagnitudePrunerConfig,
# #     OptimizationConfig,
# #     prune_weights,
# # )

# # op_config = OpMagnitudePrunerConfig(
# #     target_sparsity=0.6,
# #     weight_threshold=1024,
# # )
# # config = OptimizationConfig(global_config=op_config)
# # model_compressed = prune_weights(outputmodel, config=config)
# compressed_mlmodel.save(saved_model)




