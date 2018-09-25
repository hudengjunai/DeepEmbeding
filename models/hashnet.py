import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50

class HashNetRes50(nn.Module):
    """
    this is a hash net based on resnet50
    """
    def __init__(self,n_bit):
        super(HashNetRes50,self).__init__()
        model_resnet = resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1,
                                            self.bn1,
                                            self.relu,
                                            self.maxpool,
                                            self.layer1,
                                            self.layer2,
                                            self.layer3,
                                            self.layer4,
                                            self.avgpool)
        self.hash_layer = nn.Linear(model_resnet.fc.in_features,n_bit)
        self.hash_layer.weight.data.normal_(0,0.01)
        self.hash_layer.bias.data.fill_(0.0)
        self.activation = torch.nn.Tanh()

        self.iter_num =0
        self.gamma = 0.005
        self.step_size = 200
        self.power =0.5
        self.init_scale = 1.0
        self.scale = self.init_scale
        self.__in_features = n_bit

    def forward(self,x):
        """ the image x contains x and x' to generate similairty"""
        if self.training:
            self.iter_num +=1
        x = self.feature_layers(x)
        x = x.view(x.size(0),-1)
        y = self.hash_layer(x) # just a linear transform
        if self.iter_num % self.step_size == 0:
            self.scale = self.init_scale*math.pow((1+self.gamma*self.iter_num),self.power)
        y = self.activation(self.scale*y)
        return y

    def ouput_num(self):
        return self.__in_features

class HashLoss(nn.Module):
    def __init__(self,hash_bit):
        super(HashLoss,self).__init__()
        self.hash_bit = hash_bit


    def forward(self,x,y,sigmoid_param = 1.0,l_threshold=15.0,class_num =1.0):
        """

        :param x:
        :param y:
        :param sigmoid_param:
        :param l_threshold:  the big dot_product use the limitation
        :param class_num: the imbalance data distribution
        :return:
        """
        total_size = x.shape[0]
        x1 = x.narrow(0,0,total_size//2)
        x2 = x.narrow(0,total_size//2,total_size//2) # narrow,dimension,start,length
        y1 = y.narrow(0,0,total_size//2)
        y2 = y.narrow(0,total_size//2,total_size//2)

        similarity = torch.mm(y1,y2.t())
        dot_product = sigmoid_param * torch.mm(x1,x2.t())
        exp_product = torch.exp(dot_product)

        mask_dot = dot_product.data>l_threshold
        mask_exp = dot_product.data<=l_threshold #dot_product  比较小时候，使用log（1+exp(x)) - sij <hi,hj>

        mask_positive = similarity.data>0
        mask_negative = similarity.data<=0

        mask_dp = mask_dot & mask_positive
        mask_dn = mask_dot & mask_negative
        mask_ep = mask_exp & mask_positive
        mask_en = mask_exp & mask_negative

        dot_loss = dot_product*(1-similarity) # dot_loss 是对exp_loss在dot_product比较大时候的近似，能让dot_loss =0,
        # 在dot_product 比较大时候使用x近似log（1+exp(x))
        exp_loss = torch.log(1+exp_product) - similarity*dot_product

        loss = (torch.sum(torch.mask_select(exp_loss,mask_ep))+
                torch.sum(torch.mask_select(dot_loss,mask_dp)))*class_num + torch.sum(torch.mask_select(exp_loss,mask_en))+torch.sum(torch.mask_select(dot_loss,mask_dn))

        loss = loss /(torch.sum(mask_positive.float())*class_num +torch.sum(mask_negative.float()))
        return loss






if __name__=='__main__':

    base_resnet = HashNetRes50(n_bit=48)
    x = torch.rand((10,3,224,224))
    x = base_resnet(x)
    print(x.shape)

    #base_resnet.zero_grad()
    torch.save(nn.Sequential(base_resnet),"hashnet.pth.tar")
    print("finished")
    model = torch.load("hashnet.pth.tar")
    print(model)