import os
# import shutil
import sys
sys.path.append('cnnimageretrieval-pytorch')
import cv2
import numpy as np 
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
from tqdm import tqdm
from torchvision.transforms import functional as F
# import coremltools as ct
import torchvision
from torch.utils.model_zoo import load_url
from cirtorch.layers.pooling import MAC, SPoC, GeM, GeMmp, RMAC, Rpool
from cirtorch.networks.imageretrievalnet import init_network, extract_vectors, extract_vectors_by_arrays , extract_vectors_by_arrays2 , extract_db_array
from cirtorch.utils.general import get_data_root
def calculate_resized_dimensions(image, length_ratio):
    """ Calculate the new dimensions of the image based on the length ratio. """
    height, width = image.shape[:2]
    if width > height:
        new_width = int(length_ratio)
        new_height = int((length_ratio / width) * height)
    else:
        new_height = int(length_ratio)
        new_width = int((length_ratio / height) * width)
    return new_width, new_height
class Network(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.cuda()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    def forward(self,x):
        # x2 = F.resize(x, (400, 400))
        out2 = self.model(x.to('cuda'))   
        reshaped_tensor2 = out2.view(1, 2048)
        return reshaped_tensor2
def square_images(images, image_size=500):
    h, w = images.shape[:2]
    max_wh = max(h,w)
    if max_wh != image_size:

        if h > w:
            images = cv2.resize(images, (int(w * image_size / h), image_size))
        else:
            images = cv2.resize(images, (image_size,int(h*image_size / w)))
    tensors = np.zeros(( image_size, image_size,3))
    h, w, c = images.shape
    pad_top = int((image_size - h)/2)
    pad_left = int((image_size - w)/2)
    tensors[ pad_top:pad_top + h, pad_left: pad_left + w,:] = images
    return tensors
def get_embedding(imgpath,model):
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # newsize = calculate_resized_dimensions(img,500)
    img = square_images(img,200)
    # x = cv2.resize(img, (newsize[0], newsize[1]))
    # scale = 1/(0.226*255.0)
    # bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]
    tensor_img = torch.from_numpy(img).float()
    tensor_img = tensor_img.unsqueeze(0)
    tensor_img = tensor_img.permute(0, 3, 1, 2)
    tensor_img = tensor_img/255
    # bias_tensor = torch.tensor(bias).view(1, 3, 1, 1)
    # normalized_tensor = tensor_img * scale + bias_tensor
    torch_out = model(tensor_img)
    return torch_out.cpu().squeeze().detach().numpy()


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

dictResult = []
id = 0
# rootfolder = '/media/anlab/data-2tb/ANLAB_THUY/ToyotaAR/Dataset/NewData/SearchData/'
# for foldername in tqdm(os.listdir(rootfolder)):
#     for filename in os.listdir(rootfolder+foldername):
#         tmp = {}
#         emb = get_embedding(os.path.join(rootfolder,foldername,filename),test_model)
#         tmp['id'] = id
#         tmp['path'] = foldername+'/'+filename
#         listEmb = []
#         for val in emb:
#             listEmb.append(str(val))
#         tmp['vector'] = list(listEmb)
#         dictResult.append(tmp)
#         id+=1
# import json
# # Writing to sample.json
# with open("/media/anlab/data-2tb/ANLAB_THUY/ToyotaAR/Dataset/NewData/Vector_Cirtorch_2811_AddTransformBGR_AddNewDataV2_Augment_Square.json", "w") as outfile:
#     json.dump(dictResult, outfile)
listvec = []
listfilename = []
count = 0
for foldername in tqdm(os.listdir('/media/anlab/data-2tb/ANLAB_THUY/ImageSearcher/DataBase/geeks_image_split')):
    for filename in os.listdir('/media/anlab/data-2tb/ANLAB_THUY/ImageSearcher/DataBase/geeks_image_split/'+foldername):
        if count == 200000:
            break
        emb = get_embedding(os.path.join('/media/anlab/data-2tb/ANLAB_THUY/ImageSearcher/DataBase/geeks_image_split',foldername,filename),test_model)
        listvec.append(emb)
        listfilename.append(os.path.join(foldername,filename))
        count+=1
import faiss 
import numpy as np
import pickle
index = faiss.IndexFlatL2(2048)
index.add(np.array(listvec)) 
print(index.ntotal)
from faiss import write_index, read_index
write_index(index, "200KData.index")
index = read_index("200KData.index")
print(index.ntotal)
with open('maps_filename_200k.pickle', 'wb') as f:
    pickle.dump(listfilename, f)