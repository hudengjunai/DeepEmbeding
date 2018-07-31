from mxnet.image import *
from mxnet.gluon.data import Dataset
from mxnet.image import *
import numpy as np
import mxnet as mx
from mxnet.gluon import nn
import mxnet.gluon.data.vision.transforms as T

from data.mxdata.basic_module.basic_transform import default_transform, test_transform

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

class MxEbay(Dataset):
    """this is an mxnet edition of Ebay dataset"""
    def __init__(self,dir_root,batch_k=4,batch_size=40,is_train=True,transform =default_transform):
        self.batch_size=batch_size
        self.batch_k = batch_k
        self.root = dir_root
        self._trans = transform
        self.is_train = is_train

        self.test_image_files =[]
        self.test_labels =[]
        self.train_length = 0
        if self.is_train:
            table_name = os.path.join(self.root,'Ebay_train.txt')
            table_data = pd.read_table(table_name, header=0, delim_whitespace=True)
            min_super_id, max_super_id = min(table_data.super_class_id), max(table_data.super_class_id)
            self.super_ids = np.arange(min_super_id, max_super_id + 1)
            self.super2class = {}
            for super_id in self.super_ids:
                self.super2class[super_id] = table_data[table_data.super_class_id == super_id].class_id.tolist()

            min_class_id,max_class_id = min(table_data.class_id),max(table_data.class_id)
            self.class_ids = np.arange(min_class_id,max_class_id+1)
            self.train_length = max_class_id+1-min_class_id

            self.class2imagefiless = [[]]
            for class_id in self.class_ids:
                one_class_paths = table_data[table_data.class_id==class_id].path.tolist() # type list
                self.class2imagefiless.append(one_class_paths)
        else:
            table_name = os.path.join(self.root,'Ebay_test.txt')
            table_data = pd.read_table(table_name,header=0,delim_whitespace=True)

            self.test_image_files =  table_data.path.tolist()
            self.labels = table_data.class_id.tolist()


    def __len__(self):
        if self.is_train:
            return self.train_length
        else:
            return len(self.test_image_files)

    def sample_train_batch(self):
        batch =[]
        labels =[]
        num_groups = self.batch_size // self.batch_k  # for every sample count k
        super_id = np.random.choice(list(self.super2class.keys()), 1)[0]  # the super class id
        sampled_class = np.random.choice(self.super2class[super_id], num_groups*2, replace=False)
        for i in sampled_class:
            try:
                img_fnames = np.random.choice(self.class2imagefiless[i],
                                              self.batch_k,
                                              replace=False)
            except:
                print("class id:{0},instance count small than {1}".format(i,self.batch_k))
                continue
            batch += img_fnames.tolist()
            labels += [i]*self.batch_k
            if len(batch)>=self.batch_size:
                break
        return batch,labels


    def __getitem__(self, index):
        """get data batch like pytorch,
        only smaple same super class_id,not cross sample"""
        if self.is_train:
            imagelist =[]
            batch,labels = self.sample_train_batch()
            for file in batch:
                file_path = os.path.join(self.root,file)
                img = image.imread(file_path,to_rgb=1,flag=1)
                img = self._trans(img)
                imagelist.append(img)
            return nd.stack(*imagelist,axis=0),nd.array(labels)
        else:
            file = self.test_image_files[index]
            label = self.test_labels[index]
            img = image.imread(os.path.join(self.root,file),flag=1,to_rgb=1)
            img = self._trans(img)
            return img,label



def getEbayData(root,batch_k,batch_size):
    train_dataset = MxEbay(root,batch_k=batch_k,batch_size=batch_size,is_train=True,transform=default_transform)
    test_dataset = MxEbay(root,batch_k=batch_k,batch_size=batch_size,is_train=False,transform=test_transform)
    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=Flase,num_workers=6)
    test_loader = DataLoader(test_dataset,batch_size=test_dataset.batch_size,shuffle=False,num_workers=6)
    return train_loader,test_loader

if __name__=='__main__':
    # construct the dataset and get data in train and test mode

    train_data = MxEbay(dir_root='data/Stanford_Online_Products',\
                        batch_k=4,batch_size=40,is_train=True,\
                        transform=default_transform)
    import ipdb
    ipdb.set_trace()
    data = train_data[0]




