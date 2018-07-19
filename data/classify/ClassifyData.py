import torchvision.transforms as T
from configs import opt
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import os
import csv
import fnmatch
from PIL import Image
import numpy as np
from torch.utils.data.dataloader import default_collate

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
default_transform  = T.Compose([
         T.RandomResizedCrop(224),
         T.RandomHorizontalFlip(),
         T.ToTensor(),
         normalize,
])


#origin_dataset = ImageFolder(opt.train_classify_dir,target_transform=transform)
def my_collate_fn(batch):
    '''
    batch中每个元素形如(data, label)
    '''
    # 过滤为None的数据
    batch = list(filter(lambda x:hasattr(x[0],'size'), batch))
    if len(batch) == 0: return t.Tensor()
    return default_collate(batch) # 用默认方式拼接过滤后的batch数据

class Street2shop(Dataset):
    """ dataset split to train and test
    root is a ln -s link like this
    --root
       --bags
       --tops
       --skirts
       --hats
       --;;
       have 13 catergory consumer"""
    def __init__(self,root,train=True,persist = opt.persist,trans = default_transform):
        self.train = train
        self.root = root

        self.names_idx = {}
        self.transform = trans

        if persist is None and not os.path.exists(persist):
            folders = os.listdir(root)
            folders.sort()  # from a to x sort
            self.names_idx={fold:i for i,fold in enumerate(folders)}
            with open('data/persist.csv','w') as f:
                writer = csv.writer(f)
                for fold in folders:
                    index = self.names_idx[fold]
                    imgs = os.listdir(os.path.join(self.root,fold))
                    for img in imgs:
                        writer.writerow([fold+'/{0}'.format(img),index])

        # start to read data
        with open(persist, 'r') as f:
            reader = csv.reader(f)
            self.imgs = [row for row in reader]
        print("dataset size",len(self.imgs))

        np.random.shuffle(self.imgs)
        if self.train:
            self.imgs = self.imgs[:int(0.7 * len(self.imgs))]
        else:
            self.imgs = self.imgs[int(0.7 * len(self.imgs)):]

    def __getitem__(self, index):
        """get data and transform"""
        img_path,label = self.imgs[index]
        img_path = os.path.join(self.root,img_path)
        try:
            data = Image.open(img_path)
            if not hasattr(data,'size'):
                raise Exception("no size or data channel problem")
            if self.transform:
                data = self.transform(data)
            if not data.size(0) is 3:
                print("channel not 3,img_path is :{0}".format(img_path))
                raise Exception("channel not 3")
            return data,int(label)
        except Exception as e:
            print(e,img_path)
            img_path,label = self.imgs[0]
            data = Image.open(os.path.join(self.root,img_path))
            label = int(label)
            if self.transform:
                data = self.transform(data)
            return data,label




    def __len__(self):
        return len(self.imgs)





