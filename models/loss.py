import torch.nn as nn
from torch.nn.functional import cross_entropy
import torch

class NpairLoss(nn.Module):
    """the multi-class n-pair loss"""
    def __init__(self,l2_reg=3e-3):
        super(NpairLoss,self).__init__()
        self.l2_reg = l2_reg

    def forward(self,feature,target):
        """
        compute the feature pair loss,the first half is anchor
        the last half is pair feature
        :param feature:
        :return:
        """

        batch_size = feature.size(0)
        fa = feature[:int(batch_size/2)]
        fp = feature[int(batch_size/2):]
        logit = torch.matmul(fa,torch.transpose(fp,0,1))
        loss_sce = cross_entropy(logit,target)
        l2_loss = sum(torch.norm(feature,p=2,dim=1))/batch_size
        loss = loss_sce + self.l2_reg*l2_loss
        return loss



