# hash data from three files,contain coco,nus_wide and imangenet

from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as T
from PIL import Image
import os
import numpy as np


root_path = '/data/jh/notebooks/hudengjun/DeepEmbeding/data/hashdata'

def image_train(resize_size=256, crop_size=224):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    return  T.Compose([
        T.Resize(resize_size),
        T.RandomResizedCrop(crop_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize])

def image_test(resize_size = 256,crop_size=224):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    #start_first = 0
    start_center = (resize_size - crop_size - 1) / 2
    #start_last = resize_size - crop_size - 1

    return T.Compose([
        T.Resize(resize_size),
        PlaceCrop(crop_size,start_center,start_center),
        T.ToTensor(),
        normalize])
class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class ImageList(Dataset):
    def __init__(self,file,transform=None):
        if transform is None:
            self._transform = image_train(256,224) if 'train.txt' in file else image_test(256,224)
        else:
            self._transform = transform
        if not os.path.exists(file):
            raise Exception("file not exist")
        self.file = file

        self.images = []
        with open(self.file,'r') as f:
            for line in f.readlines():
                items =  line.strip().split(' ')
                self.images.append((items[0],np.array([int(la) for la in items[1:]],dtype=np.float32)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path,target = self.images[index]
        img = Image.open(path).convert('RGB')
        if self._transform:
            img = self._transform(img)
        return img,target

def get_hash_dataloader(dataset_name,train_batch,test_batch,database_batch):
    """
    return the double train dataset
    :param dataset:
    :return:
    """
    file_names = ['train.txt','test.txt','database.txt']
    files = [os.path.join(root_path,dataset_name,file_name) for file_name in file_names]
    datasets = [ImageList(file) for file in files]
    train1 = DataLoader(datasets[0],batch_size=train_batch,shuffle=True,num_workers=6)
    train2 = DataLoader(datasets[0],batch_size=train_batch,shuffle=True,num_workers=6)
    test = DataLoader(datasets[1],batch_size=test_batch,shuffle=False,num_workers=4)
    database = DataLoader(datasets[2],batch_size=database_batch,shuffle=False,num_workers=4)
    return train1,train2,test,database



if __name__== '__main__':
    coco_train = ImageList('/data/jh/notebooks/hudengjun/DeepEmbeding/data/hashdata/coco/train.txt')
    print("size of cocotrain",len(coco_train))
    print("start to get data",coco_train[0][0].shape,coco_train[0][1].shape)




