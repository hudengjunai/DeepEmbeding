# an implementation of mxnet in vision data dataset similar like pytorch.

from mxnet.gluon.data import DataLoader,Dataset
from mxnet import nd
from mxnet.image import imread

import os
import numpy as np
import mxnet as mx
from mxnet.gluon import nn
import mxnet.gluon.data.vision.transforms as T


class RandomCrop(nn.Block):
    def __init__(self,size):
        self.size = size
    def forward(self,x):
        return mx.image.random_crop(x,(self.size,self.size))

normalize=T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
default_transform = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(size=224,scale=(1.0,1.0),ratio=(1.0,1.0)),# just crop,not scale
    T.RandomFlipLeftRight(),
    T.ToTensor(), # last to swap  channel to c,w,h
    normalize
])

test_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    normalize
])

class CUB200Data(Dataset):
    def __init__(self,dir_path,batch_k,batch_size,is_train,transform = default_transform):
        self.dir_path = dir_path
        self.batch_k = batch_k
        self.batch_size = batch_size
        self._transform = transform
        self.is_train = is_train
        self.train_image_files = [ [] for _ in range(100)]
        self.test_images_files = [] # to store test image files
        self.test_labels = [] # to store test iamge and image label
        self.boxes = {} # to store image bounding box

        with open(os.path.join(dir_path,'images.txt'),'r') as f_img,\
            open(os.path.join(dir_path,'image_class_labels.txt'),'r') as f_label,\
            open(os.path.join(dir_path,'bounding_boxes.txt'),'r') as f_box:
            for line_img,line_label,line_box in zip(f_img,f_label,f_box):
                fname = os.path.join(self.dir_path,'images',line_img.strip().split()[-1])
                label = int(line_label.strip().split()[-1])-1
                box = [int(float(v)) for v in line_box.split()[-4:]]
                self.boxes[fname]=box

                if label<100:
                    self.train_image_files[label].append(fname)
                else:
                    self.test_images_files.append(fname)
                    self.test_labels.append(label)
        self.n_test = len(self.test_images_files)
        self.train_class_ids = list(np.arange(0,100)) #list(self.train_image_files.keys()) # get all train class id list

    def __len__(self):
        if self.is_train:
            return 200
        else:
            return self.n_test

    def __getitem__(self, index):
        """
        get the batch //batch_k for train and single for test
        """
        if self.is_train:
            image_names,labels = self.sample_train_batch()
            # get sampled order image_file names and corresponding label
            image_list,label_list=[],[]
            for img,label in zip(image_names,labels):
                image = imread(img,flag=1,to_rgb=True)
                x,y,w,h = self.boxes[img]
                image = image[y:min(y+h,image.shape[0]),x:min(x+w,image.shape[1])]
                if image.shape[2]==1:
                    print("has gray file",img)
                    image = nd.tile(image,(1,1,3))
                image =self._transform(image) # for rgb same value
                image_list.append(image)
                label_list.append(label)
            batch_data = nd.stack(*image_list,axis=0)
            batch_label = nd.array(label_list)
            return batch_data,batch_label
        else:
            img = self.test_images_files[index] # get the file name full path
            image = imread(img,flag=1,to_rgb=1)
            x,y,w,h = self.boxes[img]
            image = image[y:min(y+h,image.shape[0]),x:min(x+w,image.shape[1])]
            image = self._transform(image)

            return image,self.test_labels[index]

    def sample_train_batch(self):
        """sample batch_size//batch_k and sample small batch_k in each instance"""
        batch = []
        labels =[]
        num_groups = self.batch_size // self.batch_k
        sampleed_classes = np.random.choice(self.train_class_ids,num_groups,replace=False)
        for class_id in sampleed_classes:
            img_fname = np.random.choice(self.train_image_files[class_id],self.batch_k,replace=False)
            batch += img_fname.tolist()
            labels += [class_id]*self.batch_k
        return batch,labels


def getCUB200(data_path,batch_k,batch_size):
    train_dataset = CUB200Data(data_path,batch_k=batch_k,batch_size=batch_size,is_train=True,transform=default_transform)
    test_dataset = CUB200Data(data_path,batch_k=batch_k,batch_size=batch_size,is_train=False,transform=test_transform)
    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=6)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,num_workers=6)
    return train_loader,test_loader


if __name__=='__main__':
    import ipdb
    #ipdb.set_trace()
    train_loader, test_loader = getCUB200('data/CUB_200_2011',batch_k=5,batch_size=10)
    # for train_batch,test_batch in zip(train_loader,test_loader):
    #     print("begin to resolve data from train_loader and test_loader")
    #     ipdb.set_trace()
    #     print("data",train_batch[0][0].shape,train_batch[1][0].shape)
    #     print("test_data",test_batch[0].shape,test_batch[1].shape)
    #     break
    train_dataset = CUB200Data('data/CUB_200_2011', batch_k=5, batch_size=10, is_train=True)
    ipdb.set_trace()
    data = train_dataset[0]
    print(data)
    test_dataset = CUB200Data('data/CUB_200_2011',batch_k=5,batch_size=10,is_train=False)
    data = test_dataset[0]
    print(data)
    # for test_batch in test_loader:
    #     ipdb.set_trace()
    #     print(test_batch[0].shape,test_batch[1].shape)
    #     break




