import argparse
import fnmatch
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import sys
import numpy as np
# import onnxruntime as ort
# import onnx
import torch
# from onnx_tf.backend import prepare
# import tensorflow as tf 
from torchvision.transforms import functional as F
# from torch.utils.model_zoo import load_url
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms
# import cv2
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
class Network(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.cpu()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    def forward(self,x):
        # x1 = F.resize(x, (480, 480))
        out1 = self.model(x)   
        reshaped_tensor1 = out1.view(1, 2048)
        return reshaped_tensor1

state = torch.load(os.path.join(get_data_root(), 'networks/model_best.pth.tar'))
net_params = {}
net_params['architecture'] = state['meta']['architecture']
net_params['pooling'] = state['meta']['pooling'] 
net_params['local_whitening'] = state['meta'].get('local_whitening', False)
net_params['regional'] = state['meta'].get('regional', False)
net_params['whitening'] = state['meta'].get('whitening', True)
net_params['mean'] = state['meta']['mean']
net_params['std'] = state['meta']['std']
net_params['pretrained'] = False
net = load_network('model_best.pth.tar')
net.load_state_dict(state['state_dict'])
net.cuda()  
net.eval()

test_model = Network(net)
# test_model.eval()
# img = cv2.imread("/home/anlab/Downloads/F093B51E-58D7-473F-8202-51044F4C7F0F.png")
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# x = cv2.resize(img, (256, 256))
# tensor_img = torch.from_numpy(x).float()
# tensor_img = tensor_img.unsqueeze(0)
# tensor_img = tensor_img.permute(0, 3, 1, 2)
# example_input = torch.rand(1, 3, 256, 256)
# traced_model = torch.jit.trace(net, tensor_img)
# input = ct.TensorType(name='input_name', shape=(1, 3, 256, 256))
# mlmodel = ct.convert(traced_model, inputs=[input])
# mlmodel.save("testcoreml.mlmodel")
# # convert to onnx model
# # torch_out = test_model(tensor_img)
# traced_model = torch.jit.trace(test_model, tensor_img)
# out = traced_model(tensor_img)
# image_input = ct.ImageType(name="input",
#                            shape=tensor_img.shape)
# model = ct.convert(
#     traced_model,
#     inputs=[image_input],
#     compute_units=ct.ComputeUnit.CPU_ONLY,
# )
# model.save("ModelConvert/testcoreml.mlmodel")
# st = time.time()
# coreml_out_dict = model.predict({"input" : tensor_img})
# print(coreml_out_dict)
# print(time.time()-st)
# exit()

# import torch
# import torchvision
# import numpy as np
# import PIL
# import urllib
# import coremltools as ct

# # Load a pre-trained version of MobileNetV2 model.
# # torch_model = torchvision.models.mobilenet_v2(pretrained=True)
# # Set the model in evaluation mode.
# test_model.eval()
# # torch_model.eval()
# # Trace the model with random data.
# example_input = torch.rand(1, 3, 224, 224) 
# traced_model = torch.jit.trace(test_model, example_input)
# out = traced_model(example_input)
# # Download class labels in ImageNetLabel.txt.
# label_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
# class_labels = urllib.request.urlopen(label_url).read().decode("utf-8").splitlines()
# class_labels = class_labels[1:] # remove the first class which is background
# assert len(class_labels) == 1000
# # Set the image scale and bias for input image preprocessing.
# scale = 1/(0.226*255.0)
# bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]

# image_input = ct.ImageType(name="input_1",
#                            shape=example_input.shape,
#                            scale=scale, bias=bias)
# # Using image_input in the inputs parameter:
# # Convert to Core ML using the Unified Conversion API.
# model = ct.convert(
#     traced_model,
#     inputs=[image_input],
#     outputs=[ct.TensorType(dtype=np.float16)],
#     minimum_deployment_target=ct.target.macOS13,
#     compute_units=ct.ComputeUnit.CPU_ONLY,
# )
# # Save the converted model.
# model.save("mobilenet.mlmodel")
# # Print a confirmation message.
# print('model converted and saved')
# # Load the test image and resize to 224, 224.
# img_path = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/label_aLong/prj_label/lashinbang-server/convert_model/image/CDE_BK-1_close.jpg"
# img = PIL.Image.open(img_path)
# img = img.resize([224, 224], PIL.Image.ANTIALIAS)
# # Get the protobuf spec of the model.
# spec = model.get_spec()
# for out in spec.description.output:
#     if out.type.WhichOneof('Type') == "dictionaryType":
#         coreml_dict_name = out.name
#         break
# Make a prediction with the Core ML version of the model.
# coreml_out_dict = model.predict({"input_1" : img})
# print("coreml predictions: ")
# print("top class label: ", coreml_out_dict["classLabel"])

# coreml_prob_dict = coreml_out_dict[coreml_dict_name]

# values_vector = np.array(list(coreml_prob_dict.values()))
# keys_vector = list(coreml_prob_dict.keys())
# top_3_indices_coreml = np.argsort(-values_vector)[:3]
# for i in range(3):
#     idx = top_3_indices_coreml[i]
#     score_value = values_vector[idx]
#     class_id = keys_vector[idx]
#     print("class name: {}, raw score value: {}".format(class_id, score_value))
import coremltools as ct
# image_input = ct.ImageType(name="input_1",
#                         shape=example_input.shape,
#                         )
# # Using image_input in the inputs parameter:
# # Convert to Core ML using the Unified Conversion API.
# model = ct.convert(
#     traced_model,
#     inputs=[image_input],
#     # outputs=[ct.TensorType(dtype=np.float32)],
#     compute_units=ct.ComputeUnit.CPU_ONLY,
# )
# # Save the converted model.
# model.save("cirtorch_emb_tensor_v10.mlmodel")
# Print a confirmation message.
# print('model converted and saved')
scale = 1/(0.226*255.0)
bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]

input_shape = ct.Shape(shape=(1, 3, 300, 300))
dummy_input = torch.rand(1,3,300,300)
input_tensor = ct.ImageType(name="my_input", shape=input_shape,scale=scale, bias=bias)
traced_model = torch.jit.trace(test_model, dummy_input)

coreml_model = ct.convert(traced_model, inputs=[input_tensor], source='pytorch',convert_to="neuralnetwork")
coreml_model.save('Solar_emb_image_300x300_Nor_v1.mlmodel')
# example_input = torch.rand(1, 3, 224, 224) 
# traced_model = torch.jit.trace(net, example_input)
# out = traced_model(example_input)
# # Download class labels in ImageNetLabel.txt.
# # Set the image scale and bias for input image preprocessing.
# image_input = ct.ImageType(name="input_1",
#                         shape=example_input.shape)
# # Using image_input in the inputs parameter:
# # Convert to Core ML using the Unified Conversion API.
# model = ct.convert(
#     traced_model,
#     inputs=[image_input],
#     compute_units=ct.ComputeUnit.CPU_ONLY,
# )
# # Save the converted model.
# model.save("mymodel.mlmodel")
# # Print a confirmation message.
# print('model converted and saved')