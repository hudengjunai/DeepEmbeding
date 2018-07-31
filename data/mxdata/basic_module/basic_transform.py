import mxnet as mx
from mxnet.gluon import nn
import mxnet.gluon.data.vision.transforms as T


class RandomCrop(nn.Block):
    def __init__(self,size):
        self.size = size
    def forward(self,x):
        return mx.image.random_crop(x,(size,size))

normalize=T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
default_transform = T.Compose([
    T.Resize(256),
    RandomCrop(224),
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