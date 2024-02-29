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
state = torch.load(os.path.join(get_data_root(), 'networks/model_best.pth.tar'),map_location=torch.device('cuda'))
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
# net.cuda()  
net.eval()

test_model = Network(net)
test_model.eval()
# img = cv2.imread("/media/anlab/data-2tb/ANLAB_THUY/Serverless_Search/data-images/data-images/d05/ESE_BR-1_1.jpg")
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# x = cv2.resize(img, (400, 400))
# # # x = x/255
# scale = 1/(0.226*255.0)
# bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]

# tensor_img = torch.from_numpy(x).float()
# tensor_img = tensor_img.unsqueeze(0)
# tensor_img = tensor_img.permute(0, 3, 1, 2)
# # bias_tensor = torch.tensor(bias).view(1, 3, 1, 1)
# # normalized_tensor = tensor_img * scale + bias_tensor
# normalized_tensor = tensor_img/255
# torch_out = test_model(normalized_tensor)

# with open(r'/media/anlab/data-2tb/ANLAB_THUY/ImageSearcher/Tmp/ESE_BR-1_1_DB.txt', 'w') as fp:
#     for item in torch_out[0].detach().numpy():
#         # write each item on a new line
#         fp.write("%s, " % item)
#     print('Done')
# exit()
scale = 1/(0.226*255.0)
bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]
input_shape = ct.Shape(shape=(1, 3, 300, 300))
dummy_input = torch.rand(1,3,300,300)
input_tensor = ct.ImageType(name="my_input", shape=input_shape,scale=scale, bias=bias)
traced_model = torch.jit.trace(test_model.eval(), dummy_input)
traced_model.eval()

mlprogram = ct.convert(
    traced_model,
    minimum_deployment_target=ct.target.iOS13,
    inputs=[input_tensor],
    outputs=[ct.TensorType(name="embeddings")],
    convert_to="neuralnetwork",
    compute_units=ct.ComputeUnit.ALL,
)
spec = mlprogram.get_spec()
outputmodel = ct.models.MLModel(spec, weights_dir=mlprogram.weights_dir)
saved_model = 'ModelConvert/TestModel/Solar300_image_ALL_V1.mlmodel'
outputmodel.save(saved_model)