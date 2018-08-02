print("program begin")
from mxnet.gluon.data import DataLoader,Dataset
from mxnet import nd
from mxnet.image import imread
import os
import numpy as np
import mxnet as mx
import mxnet.gluon.data.vision.transforms as T
from collections import Counter

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


# like the
class DeepInClassFashion(Dataset):
    """
    the DeepInClassFashion dataset.read data from list_item_inshop.txt,

    """
    def __init__(self,dir_root,batch_k=4,batch_size=80,is_train = True,transform = default_transform):
        self.root = dir_root
        self.batch_k = batch_k
        self.batch_size = batch_size
        self._transform = transform
        self.is_train = is_train
        self.train_ids = set()
        self.boxes = {} # a dictionary store {key:path,value:bbox}
        self.test_images2id=[]# a list to store[(path,id),(path,id)]
        with open(os.path.join(self.root,'Anno','list_item_inshop.txt.'),'r') as f_instance:
            self.instance_count = int(f_instance.readline().strip())
            #self.instance_ids = list(f_instance.readlines())
            self.images_files = [ [] for _ in range(self.instance_count)]
        with open(os.path.join(self.root,'Anno','list_eval_partition.txt'),'r') as f_parti:
            f_instance.readline() # read pictures number
            f_instance.readline() # read information
            train_ids = []  # will use counter to duplicate checking
            for line in f_instance.readlines():
                path,item_id,status = line.strip().split(' ')
                int_id = int(item_id.split('_')[-1])
                if status is 'Train':
                    self.images_files[int_id].append(path)
                    self.train_ids.add(int_id)
                else:
                    self.test_images2id.append((path,int_id))
            # count train_ids and its distribution


        with open(os.path.join(self.root,'Anno','list_bbox_inshop.txt'),'r') as f_bbox:
            f_bbox.readline() # read count
            f_bbox.readline() # read description
            for line in f_bbox.readlines():
                list_info = line.strip().split(' ')
                path,box = list_info[0],list_info[-4:]
                self.boxes[path]=box
        #read instance ,split set,bbox data

        sub_list_test = self.images_files[list(self.test_ids)]
        self.test_len = 0
        for small_list in sub_list_test:
            self.test_len += len(small_list)
        self.build_structure()

    def build_structure(self):
        """build the folder to id structure dataset,
        construct the super class structure to select"""
        print("the img_root:%s"%(self.root))
        img_root = os.path.join(self.root,'img')
        self.super_types = {} # super_type2 ids{'men_shorts':[1,23,4,5]}
        for sexual in os.listdir(img_root):
            for clothe_type in os.listdir(os.path.join(self.root,'img',sexual)):
                ids = os.listdir(os.path.join(self.root,'img',sexual,clothe_type))
                self.super_types[sexual+'_'+clothe_type] = [int(instance_id.split('_')[-1]) for instance_id in ids]
        self.super_type_list = list(self.super_types.keys())
        self.super_type_distri = [len(self.super_types[k]) for k in self.super_types.keys()]
        self.super_type_distri /=sum(self.super_type_distri) # the distribution ,assume every id instance has 4 or five  images

    def __len__(self):
        if self.is_train:
            return len(self.train_ids)
        else:
            return self.test_len

    def sampled_batch_data(self):
        """choose an super_types,
        then choose the batch with batch_k iamges with bbox crop"""
        #sample based on the distribution
        batch =[]
        labels =[]
        num_groups = self.batch_size //self.batch_k
        super_id = np.random.choice(self.super_type_list,size=1,replace=False,\
                                    p=self.super_type_distri)
        sampled_ids = np.random.choice(self.super_types[super_id],\
                                       size=num_groups,replace=False)
        #the sampled_ids is like[1,2,5,45,23] in a super_type
        for i in sampled_ids:
            try:
                img_fname = np.random.choice(
                    self.images_files[i],
                    size=self.batch_k,
                    replace=False
                )
            except Exception as e:
                continue
            batch += img_fname
            labels += [i]*self.batch_k
        return batch,labels # format like img/man/short/id_xxxx01/01_shorts.jpg

    def __getitem__(self, index):
        if self.is_train:
            imagelist = []
            batch,labels = self.sampled_batch_data()
            for file in batch:
                file_path = os.path.join(self.root,file)
                image = imread(file_path,to_rgb=True,flag=1)
                if image.shape[2]==1:
                    print("has gray file",file)
                    image = nd.tile(image,(1,1,3))
                image = self._transform(image)
                imagelist.append(image)
            return nd.stack(*imagelist,axis=0),nd.array(labels)
        else:
            path,class_id = self.test_images2id[index]
            file_path = os.path.join(self.root,path)
            image = imread(file_path,to_rgb=True,flag=1)
            if image.shape[2]==1:
                image = nd.tile(image,(1,1,3))
            image = self._transform(image)
            return image,class_id






def getDeepInClassFashion(dir_root,batch_k,batch_size):
    """three main paramter dir,batch_k,batch_size"""
    train_data = DeepInClassFashion(dir_root=dir_root,batch_k=batch_k,batch_size=batch_size,is_train=False,\
                              transform=default_transform)
    test_data = DeepInClassFashion(dir_root=dir_root,batch_k=batch_k,batch_size=batch_size,is_train=False,\
                              transform=test_transform)
    train_loader = DataLoader(train_data,batch_size=1,shuffle=False,num_workers=6)
    test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False,num_workers=6)
    return train_loader,test_loader


class DeepCrossClassFashion(DeepInClassFashion):
    def __init__(self,dir_root,batch_k=4,batch_size=80,is_train = True,transform = default_transform):
        super(DeepCrossClassFashion,self).__init__(dir_root,batch_k,batch_size,is_train,transform)
        self.datatype='CrossClass'

    def sampled_batch_data(self):
        batch = []
        labels = []
        num_groups = self.batch_size//self.batch_k
        sampled_ids = np.random.choice(list(self.train_ids),size=num_groups,replace=False)
        for i in sampled_ids:
            try:
                img_fnames = np.random.choice(self.images_files[i],\
                                             size=self.batch_k,replace=False)
            except Exception as e:
                continue
            batch += img_fnames
            labels += [i]*self.batch_k
        return batch,labels

def getDeepCrossClassFashion(dir_root,batch_k,batch_size):
    train_data = DeepCrossClassFashion(dir_root,batch_k,batch_size=batch_size,\
                                       is_train=True,transform=default_transform)
    test_data  = DeepCrossClassFashion(dir_root,batch_k=batch_k,batch_size=batch_size,\
                                       is_train=True,transform=test_transform)
    train_loader = DataLoader(train_data,batch_size=1,shuffle=False,num_workers=6)
    test_loader = DataLoader(test_data,batch_size=batch_size,shuffle=False,num_workers=6)
    return train_loader,test_loader


if __name__ == '__main__':
    train_data = DeepInClassFashion(dir_root='data/',batch_k=4,batch_size=80,is_train=True,\
                               transform=default_transform)
    test_data = DeepCrossClassFashion(dir_root='data/',batch_k=4,batch_size=80,is_train=False,\
                                      transform=test_transform)

    data = train_data[0]
    assert(data[0][0].shape[0]==80)
    data = test_data[0]
    assert(data[0].shape[0]==80)



