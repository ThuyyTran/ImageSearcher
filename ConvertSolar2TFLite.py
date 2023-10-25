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
import onnxruntime as ort
import onnx
import torch
from onnx_tf.backend import prepare
import tensorflow as tf 
from torchvision.transforms import functional as F
# from torch.utils.model_zoo import load_url
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms
import cv2
sys.path.append('SOLAR/')
from solar_global.networks.imageretrievalnet import init_network, extract_vectors
from solar_global.datasets.testdataset import configdataset
from solar_global.utils.download import download_test
from solar_global.utils.evaluate import compute_map_and_print
from solar_global.utils.general import get_data_root, htime
from solar_global.utils.networks import load_network
from solar_global.utils.plots import plot_ranks, plot_embeddings
from torchvision import transforms
import time
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
        x = x.permute(0, 3, 1, 2)
        x1 = F.resize(x, (300, 300))
        out1 = self.model(x1)   
        reshaped_tensor1 = out1.view(1, 2048)
        x2 = F.resize(x, (480, 480))
        out2 = self.model(x2)   
        reshaped_tensor2 = out2.view(1, 2048)
        return (reshaped_tensor2 + reshaped_tensor1) / 2
  
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
test_model.eval()
img = cv2.imread("/home/anlab/Downloads/F093B51E-58D7-473F-8202-51044F4C7F0F.png")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
x = cv2.resize(img, (480, 480))
tensor_img = torch.from_numpy(x).float()
tensor_img = tensor_img.unsqueeze(0)
# convert to onnx model
torch_out = test_model(tensor_img)
# print('Out Pytorch',torch_out)
# exit()
onnx_path = "ModelConvert/1910_Solar_fl32_480x480_300x300_Withoutnor.onnx"
torch.onnx.export(test_model,
            tensor_img,
            onnx_path,
            verbose=True,
            input_names=["images"],
            output_names=["outputs"],
            export_params=True,
            opset_version = 10
            )
# Checker
onnx_model = onnx.load( onnx_path)
onnx.checker.check_model(onnx_model)
tf_path = 'ModelConvert/SOLAR_tf_1910_480x480_300x300_Withoutnor'
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
tf_rep = prepare(onnx_model)  #Prepare TF representation
tf_rep.export_graph(tf_path)  #Export the model

# convert to tf lite
tf_lite_path = 'ModelConvert/1910_Solar_fl32_480x480_300x300_Withoutnor.tflite'
converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)

# If want Optimize convert float16
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter.target_spec.supported_types = [tf.float16]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# done convert float16
tflite_model  = converter.convert()
with open(tf_lite_path, 'wb') as f:
    f.write(tflite_model)

#test model tflite
tflite_model_path = tf_lite_path
# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Test the model on random input data
input_shape = input_details[0]['shape']
print(tensor_img.shape)
input_data = np.array(tensor_img, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
st = time.time()
interpreter.invoke()
print(time.time()-st)
# get_tensor() returns a copy of the tensor data
# use tensor() in order to get a pointer to the tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)