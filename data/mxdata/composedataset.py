import mxnet.gluon.data.vision.transforms as T
from mxnet.gluon.data import DataLoader,Dataset
from mxnet import nd
import numpy as np
import os
from mxnet.image import imread
import pandas as pd


normalize=T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
default_transform = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(224),
    T.RandomFlipLeftRight(),
    T.ToTensor(),
    normalize
])

class ComposeDataSet(Dataset):
    """ an Ebay and DeepFashion Composite Dataset to metric learning"""
    def __init__(self,ebay_dir,fashion_dir,batch_k,batch_size,is_train,transform=default_transform):
        self.ebay_dir = ebay_dir
        self.fashion_dir = fashion_dir
        self.batch_k = batch_k
        self.batch_size = batch_size
        self.is_train = is_train
        self._transform = transform

        #begin to resolve ebay data

        if self.is_train:
            #start ebay data
            table_name = os.path.join(self.ebay_dir,'Ebay_train.txt')
            table_data = pd.read_table(table_name,header=0,delim_whitespace=True)
            min_super_id, max_super_id = min(table_data.super_class_id), max(table_data.super_class_id)

            #this is the super id for ebaydata
            self.super_ids = np.arange(min_super_id, max_super_id + 1)
            self.super2class = {} #store a dict for {super_id:[class_id1,class_id2]}
            for super_id in self.super_ids:
                self.super2class[super_id] = table_data[table_data.super_class_id == super_id].class_id.tolist()

            min_class_id, max_class_id = min(table_data.class_id), max(table_data.class_id)
            self.class_ids = list(np.arange(min_class_id, max_class_id + 1))
            self.train_length = max_class_id + 1 - min_class_id
            self.super_id_dist = [len(v) for k, v in self.super2class.items()]
            for class_id in self.class_ids:
                one_class_paths = table_data[table_data.class_id == class_id].path.tolist()  # type list
                self.class2imagefiless.append(one_class_paths)

            #Process deepfashion data
            extract_super_ids_to_class_ids
        else:
