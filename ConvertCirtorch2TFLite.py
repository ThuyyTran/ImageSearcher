import os
import shutil
import sys
sys.path.append('../')

import cv2
import numpy as np 
import onnx
import torch
import tensorflow as tf 
from PIL import Image
from torchvision.models import *
from onnx_tf.backend import prepare
import torch.nn as nn
import torch.quantization
from extract_cnn import *
import onnxruntime as ort
import onnx
from torchvision.transforms import functional as F
from torch.utils.model_zoo import load_url
from cirtorch.layers.pooling import MAC, SPoC, GeM, GeMmp, RMAC, Rpool
from cirtorch.networks.imageretrievalnet import init_network, extract_vectors, extract_vectors_by_arrays , extract_vectors_by_arrays2 , extract_db_array
from cirtorch.utils.general import get_data_root

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
        x2 = F.resize(x, (400, 400))
        out2 = self.model(x2)   
        reshaped_tensor2 = out2.view(1, 2048)
        return (reshaped_tensor2 + reshaped_tensor1) / 2


if __name__ == '__main__':

    PRETRAINED = {
    'rSfM120k-tl-resnet50-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'rSfM120k-tl-resnet101-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'rSfM120k-tl-resnet152-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet152-gem-w-f39cada.pth',
    'gl18-tl-resnet50-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'gl18-tl-resnet101-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
    'gl18-tl-resnet152-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet152-gem-w-21278d5.pth',
}

    useRmac=False
    transform_ratio = 300

    network = 'rSfM120k-tl-resnet101-gem-w'
    state = load_url(PRETRAINED[network], model_dir=os.path.join(get_data_root(), 'networks'))

    net_params = {}
    net_params['architecture'] = state['meta']['architecture']
    net_params['pooling'] = state['meta']['pooling']
    net_params['local_whitening'] = state['meta'].get('local_whitening', False)
    net_params['regional'] = state['meta'].get('regional', False)
    net_params['whitening'] = state['meta'].get('whitening', True)

    net_params['mean'] = state['meta']['mean']
    net_params['std'] = state['meta']['std']
    net_params['pretrained'] = False

    net = init_network(net_params)
    net.load_state_dict(state['state_dict'])
    if useRmac:
        net.pool = RMAC(3)

    net.cuda()
    net.eval()

    test_model = Network(net)
    test_model.eval()

    img = cv2.imread("/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/label_aLong/prj_label/lashinbang-server/convert_model/image/CDE_BK-1_close.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    x = cv2.resize(img, (400, 400))
    tensor_img = torch.from_numpy(x).float()
    tensor_img = tensor_img.unsqueeze(0)
    onnx_path = "1910_400x400_300x300_cirtorch_fl32_resnet101_4D_WithoutNorm.onnx"
    # convert to onnx model
    torch_out = test_model(tensor_img)
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
    # convert onnx to tf
    tf_path = 'cirtorch_tf_1910_400x400_300x300_resnet101_WithoutNorm'
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    tf_rep = prepare(onnx_model)  #Prepare TF representation
    tf_rep.export_graph(tf_path)  #Export the model

    # convert to tf lite
    tf_lite_path = '1910_400x400_300x300_cirtorch_fl16_resnet101_WithoutNorm.tflite'
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
    # print(x.shape)
    # input_data = np.array(x, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], tensor_img)
    interpreter.invoke()
    # get_tensor() returns a copy of the tensor data
    # use tensor() in order to get a pointer to the tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)