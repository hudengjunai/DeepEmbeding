import mxnet.gluon.data.vision.transforms as T
from mxnet.gluon.data import DataLoader,Dataset
from mxnet import nd
import numpy as np
import os
from mxnet.image import imread


normalize=T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
default_transform = T.Compose([
    T.Resize(256),
    T.RandomResizedCrop(224),
    T.RandomFlipLeftRight(),
    T.ToTensor(),
    normalize
])

class Compose