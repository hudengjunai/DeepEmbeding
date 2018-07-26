import torch
from torch.utils.data import Dataset
from PIL import Image

import torchvision.transforms as T
import numpy as np
import os
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
default_transform  = T.Compose([
         T.Resize(256),
         T.RandomCrop(224),
         T.RandomHorizontalFlip(),
         T.ToTensor(),
         normalize,
])

class CUB200DataSet(Dataset):
    """
    the cub200 bird dataset,dataset description:
    200 catagory bird, 100 for train ,100 for test ,each catagory hase 60 images

    """
    def __init__(self,data_path,batch_k=5,batch_size=70,is_train=True,transform = default_transform):
        self.is_train = is_train
        self.batch_k = batch_k                #sample numbers in every calsses,for example,5
        self.batch_size = batch_size          #the whole batch samples to fetch ,for example,70,so the sampled classes is 12
        self.train_image_files = [[]for _ in range(100)]
        self.test_image_files =[]
        self.test_labels =[]
        self.boxes = {}
        self.transform = transform

        with open(os.path.join(data_path,'images.txt'),'r') as f_img,\
            open(os.path.join(data_path,'image_class_labels.txt'),'r') as f_label,\
            open(os.path.join(data_path,'bounding_boxes.txt'),'r') as f_box:
            for line_img,line_label,line_box in zip(f_img,f_label,f_box):
                fname = os.path.join(data_path,'images',line_img.strip().split()[-1])
                label = int(line_label.strip().split()[-1])-1
                box = [int(float(v)) for v in line_box.split()[-4:]]
                self.boxes[fname]=box

                if label<100:
                    self.train_image_files[label].append(fname)
                else:
                    self.test_image_files.append(fname)
                    self.test_labels.append(label)

        self.n_test = len(self.test_image_files)

    def __getitem__(self, index):
        """
        get data item in train dataset,all test dataset
        :param index:  the index of training or test of sample
        :return: return the origin image data and labels based on sample method,
                  search batch/batch_k classes ,every class,choose batch_k iamges to compound a batch
        """
        if self.is_train:
            #get train batch
            images = []
            labels = []
            num_groups = self.batch_size//self.batch_k
            sampled_classes = np.random.choice(100,num_groups,replace=False)
            for class_id in sampled_classes:
                img_fnames = np.random.choice(self.train_image_files[class_id],self.batch_k,replace=False)
                for file_path in img_fnames:
                    x,y,w,h = self.boxes[file_path]
                    img = Image.open(file_path).convert('RGB').crop((x,y,x+w,y+h))
                    try:
                        img_tensor = self.transform(img)
                        images.append(img_tensor)
                        labels.append(class_id)
                    except Exception as e:
                        print(file_path)
                        break

            batch_data = torch.stack(images,dim=0)                   # from list of tensor to batch tensor
            label_data = torch.tensor(np.array(labels,dtype=np.int32))   # from list to tensor
            return batch_data,label_data
        else:
            #get one sample
            image = Image.open(self.test_image_files[index]).convert('RGB')
            label = self.test_labels[index]
            if self.transform:
                image = self.transform(image)
            return image,label

    def __len__(self):
        if self.is_train:
            return 200  #
        else:
            return self.n_test # will return all test_image_files



if __name__=='__main__':
    import ipdb
    ipdb.set_trace()
    dataset = CUB200DataSet(data_path='data/cub200_2011/CUB_200_2011/')
    data = dataset[1]
    print(type(data))
    print(data[1])