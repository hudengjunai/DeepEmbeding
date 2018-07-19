import torch
import torchvision
import torch.nn as nn

import torch.nn.functional as F
from torchvision.models.inception import  inception_v3
from .vgg_classify import BaseModule
from collections import OrderedDict
class ModGoogLeNet(BaseModule):
    """the change the head from googlenet"""
    def __init__(self,embeding_size=512,with_drop=False):
        super(ModGoogLeNet,self).__init__()
        basic_model = inception_v3(pretrained=True, transform_input=False)
        basic_model.aux_logits=False
        feature = list(basic_model.named_children())
        def aux(name_module):
            return 'AuxLogits' not in name_module[0]

        del feature[-1]
        feature = filter(aux, feature) #generator
        feature = [m for m in feature]
        self.level1_2 = nn.Sequential(OrderedDict(feature[0:3]))
        self.level_3_4 = nn.Sequential(OrderedDict(feature[3:5]))
        self.level_5_6 = nn.Sequential(OrderedDict(feature[5:13]))
        self.level_7 = nn.Sequential(OrderedDict(feature[13:16]))
        self.fc = nn.Linear(in_features=2048,out_features=embeding_size)
        self.model_name = 'DMLGoogle'
        self.with_drop = with_drop

    def freeze_model(self,level=5):
        """

        :param level: the freeze level,all the model split in (
        Conv2d_1a_3x3
        Conv2d_2a_3x3,Conv2d_2b_3x3,
        Conv2d_3b_1x1,Conv2d_4a_3x3,
        Mixed_5b,Mixed_5c,Mixed_5d,
        Mixed_6a,Mixed_6b,Mixed_6c,Mixed_6d,Mixed_6e,AuxLogits
        Mixed_7a,Mixed_7b,Mixed_7c

        :return:
        """
        for i,(name,module) in enumerate(self.basic_model.named_children()):
            if i<10 and int(name.split('_')[1][0])<=level:
                for param in module.parameters():
                    param.requried_grad = False


    def forward(self,x,normalize=False):
        """
        forward data data shape (32,2,227,227)
        :param x: torch.tensor
        :return: feature embeding
        """
        x = self.level1_2(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.level_3_4(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.level_5_6(x)
        x = self.level_7(x)

        x = F.avg_pool2d(x, kernel_size=x.size(-1)) #default 8*8,another 5*5
        # 1 x 1 x 2048
        if self.with_drop:
            x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        embeding = self.fc(x)
        return embeding


