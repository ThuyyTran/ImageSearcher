import os
# import shutil
import sys
sys.path.append('cnnimageretrieval-pytorch')
# import cv2
# import numpy as np 
# import onnx
import torch
# import tensorflow as tf 
# from PIL import Image
from torchvision.models import *
# from onnx_tf.backend import prepare
import torch.nn as nn
import torch.quantization
# from extract_cnn import *
# import onnxruntime as ort
# import onnx
from torchvision.transforms import functional as F
import coremltools as ct
import torchvision
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
        # x2 = F.resize(x, (400, 400))
        out2 = self.model(x)   
        reshaped_tensor2 = out2.view(1, 2048)
        return reshaped_tensor2


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

    # net.cuda()
    net.eval()

    test_model = Network(net)
    test_model.eval()

    # img = cv2.imread("/home/anlab/Downloads/F093B51E-58D7-473F-8202-51044F4C7F0F.png")
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # x = cv2.resize(img, (224, 224))
    # tensor_img = torch.from_numpy(x).float()
    # tensor_img = tensor_img.unsqueeze(0)
    # tensor_img = tensor_img.permute(0, 3, 1, 2)
    # onnx_path = "test.onnx"
    # torch_out = test_model(tensor_img)
    # torch.onnx.export(test_model,
    #             tensor_img,
    #             onnx_path,
    #             verbose=True,
    #             input_names=["images"],
    #             output_names=["outputs"],
    #             export_params=True,
    #             opset_version = 10
    #             )

    # import coremltools as ct
    # # Convert from ONNX to Core ML
    # model  = ct.converters.onnx.convert(model=onnx_path)
    # model.save("testcoreml.mlmodel")
    # example_input = torch.rand(1, 3, 224, 224) 
    # traced_model = torch.jit.trace(test_model, example_input)
    # out = traced_model(example_input)
    # model = ct.convert(
    # traced_model,
    # inputs=[ct.TensorType(shape=example_input.shape)]
    # )
    # # Save the converted model.
    # model.save("newmodel.mlmodel")
    # model = torchvision.models.mobilenet_v2()
    # model.eval()
    # print()
    # model = torchvision.models.mobilenet_v2()
    # model.eval()
    # # example_input = torch.rand(1, 3, 256, 256)
    # # traced_model = torch.jit.trace(model, example_input)

    # example_input = torch.rand(1, 3, 224, 224) 
    # traced_model = torch.jit.trace(model, example_input)
    # image_input = ct.ImageType(name="input_1",
    #                        shape=example_input.shape,
    #                       )
    # out = traced_model(example_input)
    # # input = ct.TensorType(name='input_name', shape=(1, 3, 256, 256))
    # mlmodel = ct.convert(traced_model, inputs=[image_input],outputs=[ct.TensorType()])
    # mlmodel.save("mobilenet1.mlmodel")

    # Load a pre-trained version of MobileNetV2 model.
    # torch_model = torchvision.models.mobilenet_v2(pretrained=True)
    # Set the model in evaluation mode.
    # torch_model.eval()
    # Trace the model with random data.
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
    # cvmodel  = ct.converters.onnx.convert(model = onnx_path,minimum_ios_deployment_target='13')
    # cvmodel.save('Mymodel.mlmodel')
    # example_input = torch.rand(1, 3, 224, 224) 
    # traced_model = torch.jit.trace(test_model, example_input)
    # out = traced_model(example_input)
    # Download class labels in ImageNetLabel.txt.
    # Set the image scale and bias for input image preprocessing.
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
    print('model converted and saved')
    scale = 1/(0.226*255.0)
    bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]

    input_shape = ct.Shape(shape=(1, 3, 300, 300))
    dummy_input = torch.rand(1,3,300,300)
    input_tensor = ct.ImageType(name="my_input", shape=input_shape,scale=scale, bias=bias)
    traced_model = torch.jit.trace(test_model, dummy_input)

    coreml_model = ct.convert(traced_model, inputs=[input_tensor], source='pytorch')
    coreml_model.save('cirtorch_emb_image_300x300_Nor.mlmodel')