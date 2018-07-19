import torch
import torch.nn as nn
import time
from torchvision.models import vgg16_bn

class BaseModule(nn.Module):
    """model save and load"""
    def __init__(self):
        super(BaseModule,self).__init__()
        self.model_name = str(type(self))
        self.model_name='basemodel'

    def load(self,path):
        """
        加载模型
        :param path: reload model path
        :return: None
        """
        self.load_state_dict(torch.load(path))

    def save(self,name=None):
        """default modelname and time"""
        if name is None:
            prefix = 'checkpoints/'+self.model_name+'_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

class VggClassify(BaseModule):
    """a model viaries from vgg_16"""
    def __init__(self,num_classes):
        super(VggClassify, self).__init__()
        vgg16_model = vgg16_bn(pretrained=False)
        features,classifier = vgg16_model.features,vgg16_model.classifier
        classifier = list(classifier)
        del classifier[-1]
        classifier.append(nn.Linear(4096,num_classes))
        self.features = features
        self.classifier = nn.Sequential(*classifier)
        self.model_name = 'vgg_bn'
    def forward(self,x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output

if __name__=='__main__':
    model = BaseModule()
    model.save()



