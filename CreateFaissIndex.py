import json
import numpy as np
import os
import faiss   
from faiss import write_index, read_index
# make faiss available
vectorfile = "/media/anlab/data-2tb/ANLAB_THUY/ImageSearcher/DataBase/json/Vector_Solar_Mitsui_Original_square_L2.json"
indexfile = "/media/anlab/data-2tb/ANLAB_THUY/ImageSearcher/DataBase/json/Index_Vector_Solar_Mitsui_Original_square_L2.index"
mapsID2Jsonfile = "/media/anlab/data-2tb/ANLAB_THUY/ImageSearcher/DataBase/json/MapsIndex_Vector_Solar_Mitsui_Original_square_L2.json"
listProductFile = '/media/anlab/data-2tb/ANLAB_THUY/ImageSearcher/DataBase/json/ListProductNew_Augment.json'
f = open(vectorfile)
database = json.load(f)
allvector = []
mapsIndex2Filename = {}
for i in range(len(database)):
    numberlist = [float(item) for item in database[i]['vector']]
    allvector.append(np.array(numberlist))
    mapsIndex2Filename[i] = os.path.join(database[i]['path'])
index_flat_l2 = faiss.IndexFlatL2(len(allvector[0]))
index = faiss.IndexIDMap2(index_flat_l2)
vectors = np.array(allvector)  # Your vectors
ids = np.array(range(len(vectors)))  # Example IDs
index.add_with_ids(vectors, ids)
# print(index.is_trained)
# index.add(np.array(allvector))                  # add vectors to the index
print(index.ntotal)
write_index(index, indexfile)
listProduct = []
mapsfilename2id = {}
id = 0
rootfolder = '/media/anlab/data-2tb/ANLAB_THUY/ImageSearcher/DataBase/Database_ORIGINAL/'
for foldername in os.listdir(rootfolder):
    tmplist = {'id':id,'Product_Name':foldername,'Product_Detail':foldername,'List_Image':[]}
    for filename in os.listdir(rootfolder+foldername):
        tmplist['List_Image'].append(os.path.join(foldername,filename))
        mapsfilename2id[os.path.join(foldername,filename)] = id
    id+=1
    listProduct.append(tmplist)
with open(listProductFile, 'w') as fp:
    json.dump(listProduct, fp)
mapsIndex2Filename = {}
for i in range(len(database)):
    mapsIndex2Filename[i] = {'path':os.path.join(database[i]['path']),'id':mapsfilename2id[os.path.join(database[i]['path'])]}
with open(mapsID2Jsonfile, 'w') as fp:
    json.dump(mapsIndex2Filename, fp)
    