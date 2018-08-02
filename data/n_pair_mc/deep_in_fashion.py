# this is an implementation of pytorch deep_fashion_in dataset,
# aim to train an multi-class-n-pair model as base line

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


class DeepFashionTorch(Dataset):
