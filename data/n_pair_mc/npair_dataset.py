import torchvision.transforms as T
#from configs import opt
from torch.utils.data import Dataset
import os
import csv
import fnmatch
from PIL import Image
import numpy as np
import pandas as pd
import torch

#normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
normalize = T.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
default_transform  = T.Compose([
         T.Resize(256),
         T.RandomCrop(227),
         T.RandomHorizontalFlip(),
         T.ToTensor(),
         normalize,
])

class EbayDataset(Dataset):
    """this is an implementation of n-pair mc dataset from paper
    Improved Deep Metric Learning with Multi-class N-pair Loss """
    def __init__(self,dir_root=None,
                 train=True,
                 batch_size=32,
                 trans = default_transform):
        """

        :param dir_root: online product dir
        :param train: if train get data with pair else return each image with label
        :param persist: Ebay_test.txt and Ebay_train.txt to read basic data
        :param batch_size: pair*2 ,if batch_size =32,the pairsize is 16
        :param trans: data transformation
        """
        self.batch_size = batch_size
        self.root = dir_root
        self.train = train
        self.transform = trans

        file_name =os.path.join(self.root,'Ebay_train.txt')
        if not self.train:
            file_name = os.path.join(self.root, 'Ebay_test.txt')
        self.data = pd.read_table(file_name,header=0,delim_whitespace=True)
        min_super_id,max_super_id = min(self.data.super_class_id),max(self.data.super_class_id)
        self.super_ids =np.arange(min_super_id,max_super_id+1)
        self.super2class={}
        for super_id in self.super_ids:
            self.super2class[super_id]=self.data[self.data.super_class_id==super_id].class_id.tolist()

        self.all_class = list(set(self.data.class_id.tolist()))
        self.classid2imageid = {}
        for class_id in self.all_class:
            group_image_id = self.data[self.data.class_id==class_id].image_id.tolist()
            if len(group_image_id)>=2:
                self.classid2imageid[class_id]=group_image_id #one group must have more than 2 images
        self.image_nums = self.data.image_id.count()

    def __len__(self):
        """the lengh and data loader recycle size"""
        if self.train:
            return len(self.all_class) # 11318
        else:
            return self.image_nums

    def __getitem__(self, index):
        """get pair size pair data with index
        when using dataloader ,the batchsize is always 1
        for train model:
            the index is class_id ,so this will select a batch of different class type to construct a n-pair
        for test model:
            the index is image_id,so this will get one picture with it's image_id and class_id,the extracted feature will send to cluster
        """
        if self.train:
            class_id = self.all_class[index]
            super_id = self.data[self.data.class_id==1].super_class_id[0]
            anchor_class=[]
            anchor_class.append(class_id)
            innder_count = int(0.9* self.batch_size//2) # image pair of different class in same super class

            inner_class = np.random.choice(self.super2class[super_id], innder_count, False) # in same super class choose most
            anchor_class.extend(inner_class)
            anchor_class = list(set(anchor_class))# duplicate repeate

            outer_count = self.batch_size//2 - len(anchor_class)
            outer_class = np.random.choice(self.super_ids,outer_count,True)
            for outer_id in outer_class:
                anchor_class.extend(np.random.choice(self.super2class[outer_id],1))

            #from each anchor_class,select the anchor image and the postive image
            image_id =[]
            for anchor_id in anchor_class:
                select = np.random.choice(self.classid2imageid[anchor_id],2,False)
                image_id.extend(select)


            anchor_path = self.data[self.data.image_id.isin(image_id)][['image_id', 'path']]
            anchor_path.sort_index(0) # sort by the first colum index id
            # to stack image in a to construct to one bulk. first construct 32 image to a numpy ndarray,
            tensor_list=[]
            tensor_p=[]
            jump = False
            for i,image_path in enumerate(anchor_path.path):
                image = Image.open(os.path.join(self.root,image_path)).convert('RGB')
                if self.transform:
                    data = self.transform(image)
                if i%2==0:
                    if data.size(0)<3:   # the anchor image channel not 3
                        jump = True  # jump the next image
                        continue
                    jump = False
                    tensor_list.append(data)
                else:
                    if jump:
                        continue
                    if data.size(0)<3: # the pair iamge channel not 3
                        tensor_list.pop(-1) # delete the last one in tensor_list
                        continue
                    tensor_p.append(data)


            tensor_list.extend(tensor_p)
            #print("tensor dataset",len(tensor_list))
            batch_tensor = torch.stack(tensor_list,dim=0)
            return batch_tensor
        else:
            item = self.data.loc[index]
            image_path = item['path']
            image_id = int(item['image_id'])
            image_class= int(item['class_id'])
            default_path = self.data.loc[0]['path']
            image = Image.open(os.path.join(self.root, image_path)).convert('RGB')
            if self.transform:
                data = self.transform(image)
                if data.size(0)<3:
                    image = Image.open(os.path.join(self.root, default_path))
                    data = self.transform(image)
                    image_id =0
                    image_class =0
            return data,image_id,image_class



if __name__=='__main__':
    """ to test the dataset"""
    import ipdb
    ipdb.set_trace()
    root = '/data/jh/notebooks/hudengjun/DML/deep_metric_learning/lib/online_products/Stanford_Online_Products/'
    dataset = EbayDataset(dir_root=root)
    data = dataset[0]
    print(type(data))

    test_dataset  = EbayDataset(dir_root=root,train=False)
    data = test_dataset[0]
    print(data)





