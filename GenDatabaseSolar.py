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
from tqdm import tqdm
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
# import coremltools
# import coremltools as ct

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

    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(list1, list2)]))
class Network(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.cuda()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    def forward(self,x):
        x1 = F.resize(x, (480, 480))
        out1 = self.model(x1)   
        reshaped_tensor1 = out1.view(1, 2048)
        # x2 = F.resize(x, (300, 300))
        # out2 = self.model(x2)   
        # reshaped_tensor2 = out2.view(1, 2048)
        return reshaped_tensor1
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
    # x = cv2.resize(img, (newsize[0], newsize[1]))
    # tensor_img = torch.from_numpy(x).float()
    # scale = 1/(0.226*255.0)
    # bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]
    img = square_images(img,100)
    tensor_img = torch.from_numpy(img).float()
    tensor_img = tensor_img.unsqueeze(0)
    tensor_img = tensor_img.permute(0, 3, 1, 2)
    tensor_img = tensor_img/255
    # bias_tensor = torch.tensor(bias).view(1, 3, 1, 1)
    # normalized_tensor = tensor_img * scale + bias_tensor
    torch_out = model(tensor_img.to('cuda'))
    return torch_out.cpu().squeeze().detach().numpy()
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
dictResult = []
id = 0
rootfolder = '/media/anlab/data-2tb/ANLAB_THUY/ImageSearcher/DataBase/Database_ORIGINAL/'
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
# with open("/media/anlab/data-2tb/ANLAB_THUY/ImageSearcher/DataBase/json/Vector_Solar_Mitsui_Original_square.json", "w") as outfile:
#     json.dump(dictResult, outfile)
listvec = []
listfilename = []
count = 0
for foldername in tqdm(os.listdir('/media/anlab/data-2tb/ANLAB_THUY/ImageSearcher/DataBase/geeks_image_split')):
    for filename in os.listdir('/media/anlab/data-2tb/ANLAB_THUY/ImageSearcher/DataBase/geeks_image_split/'+foldername):
        if count == 200000:
            break
        emb = get_embedding(os.path.join('/media/anlab/data-2tb/ANLAB_THUY/ImageSearcher/DataBase/geeks_image_split',foldername,filename),test_model)
        # # print(emb)
        # import base64
        # # base64_bytes = base64.b64encode(cv2.imread(os.path.join('/media/anlab/data-2tb/ANLAB_THUY/ImageSearcher/DataBase/geeks_image_split',foldername,filename)))
        # from PIL import Image
        # import io
        # import base64
        # image_array = cv2.imread(os.path.join('/media/anlab/data-2tb/ANLAB_THUY/ImageSearcher/DataBase/geeks_image_split',foldername,filename))
        # image = Image.fromarray(image_array)
        # buffer = io.BytesIO()
        # image.save(buffer, format="JPEG")  # You can change JPEG to PNG if you prefer
        # image_bytes = buffer.getvalue()
        # encoded_image = base64.b64encode(image_bytes)
        # encoded_image_str = encoded_image.decode('utf-8') 
        # with open(r'imagebase64.txt', 'w') as fp:
        #     fp.write("," + str(encoded_image_str))
        # decoded_image_data = base64.b64decode(encoded_image_str)

        # # Read the image data from a bytes buffer
        # image = Image.open(io.BytesIO(decoded_image_data))

        # # Convert the image back to a NumPy array
        # decoded_image_array = np.array(image)
        # cv2.imwrite('test.png',decoded_image_array)
        # with open(r'Testvector.txt', 'w') as fp:
        #     for item in np.array(emb):
        #         fp.write("," + str(item))
        # exit()
        listvec.append(emb)
        listfilename.append(os.path.join(foldername,filename))
        count+=1
import faiss 
import numpy as np
index = faiss.IndexFlatL2(2048)
index.add(np.array(listvec)) 
print(index.ntotal)
from faiss import write_index, read_index
write_index(index, "200KData.index")
index = read_index("200KData.index")
print(index.ntotal)
with open('maps_filename_200k.pickle', 'wb') as f:
    pickle.dump(listfilename, f)