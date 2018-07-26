import argparse
import time
import logging
import numpy as np
import torch

from torch.utils.data import DataLoader
import os
from data import CUB200DataSet
from models import Margin_Loss,SampleModel


parser = argparse.ArgumentParser(description="train a margin based loss model")
parser.add_argument('--data_path',type=str,default="data/cub200_2011",
                    help='path of the cub_data')
parser.add_argument('--embed_dim',type=int,default=128,
                    help='dimensionality of image embeding,times of 8')
parser.add_argument('--batch_size',type=int,default=70,
                    help='training batch size per device')
parser.add_argument('--batch_k',type=int,default=5,
                    help='number of images per class in a batch,can be divided by batch_size')
parser.add_argument('--gpu_id',type=str,default='0',
                    help='the gpu_id of the runing batch')
parser.add_argument('--epochs',type=int,default=100,
                    help='number of training epochs,default is 100')
parser.add_argument('--optimizer',type=str,default='adam',
                    help='optimizer,default is adam')
parser.add_argument('--lr',type=float,default=0.0001,
                    help='learning rate of the resnet and dense layer')
parser.add_argument('--lr_beta',type=float,default=0.1,
                    help='learning rate for the beta in margin based loss')
parser.add_argument('--margin',type=float,default=0.2,
                    help='margin for the margin based loss,default is 0.2')
parser.add_argument('--beta',type=float,default=1.2,
                    help='the class specific beta parameter')
parser.add_argument('--nu',type=float,default=0.0,
                    help='regularization parameter for beta,default is 0')
parser.add_argument('--steps',type=str,default='30,50,100,300',
                    help='epochs to updata learning rate')
parser.add_argument('--wd',type=float,default=0.0001,
                    help='weight decay rate,default is 0.00001')
parser.add_argument('--seed',type=int,default=123,
                    help='random seed to use,default=123')
parser.add_argument('--factor',type=float,default=0.5,
                    help='learning rate schedule factor,default is 0.5')
parser.add_argument('--print_freq',type=int,default=20,
                    help='print the accumulate loss for training process')
parser.add_argument('--debug',action='store_true',default=False)


opt = parser.parse_args()
logging.info(opt)
torch.random.manual_seed(opt.seed)
np.random.seed(opt.seed)
batch_size = opt.batch_size
os.environ['CUDA_VISIBLE_DEVICES']=opt.gpu_id
steps = [int(step) for step in opt.steps.split(',')]


def train():
    """
    train the margin based loss model
    :return:
    """
    # prepare for data for loader
    train_data = CUB200DataSet(data_path='data/cub200_2011/CUB_200_2011',batch_k=opt.batch_k,batch_size = opt.batch_size,is_train=True)
    test_data = CUB200DataSet(data_path='data/cub200_2011/CUB_200_2011',is_train=False)

    train_loader = DataLoader(train_data,batch_size=1,shuffle=False,num_workers=6)
    test_loader = DataLoader(test_data,batch_size=60,shuffle=False,num_workers=6)

    #begin to set model loss,optimizer,lr_rate, lr_schedule
    model = SampleModel(embeding_dim=opt.embed_dim)
    beta = torch.tensor(np.ones(100)*opt.beta, requires_grad=True,dtype=torch.float32)

    loss_criterion = Margin_Loss(batch_k=opt.batch_k,\
                                 margin=opt.margin,nu=opt.nu) # set loss function for this model

    conv_params = []
    non_conv_param =[]
    for name,param in model.base_model.named_parameters():
        if 'conv' in name:
            conv_params.append({'params':param,'lr':opt.lr*0.01})
        else:
            non_conv_param.append({'params':param,'lr':opt.lr})
    total_param =[]
    total_param.append({'params':model.dense.parameters(),'lr':opt.lr})
    total_param.extend(conv_params)
    total_param.extend(non_conv_param)
    optimizer = torch.optim.Adam(total_param,lr=opt.lr,weight_decay=opt.wd)
    optimizer_beta = torch.optim.SGD([{'params':beta}],lr=opt.lr_beta,momentum= 0.9)


    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                       milestones=steps,gamma=opt.factor)

    if int(opt.gpu_id)>=0:
        model = model.cuda()  # the loss function has paramter to convey to cuda
        beta = beta.cuda() # the beta parameter has parameter to stored in cuda
        loss_criterion = loss_criterion.cuda()  # the loss criterion has compute in cuda
        loss_criterion.convert_param(to_cuda=True)

    # begin to fetch data and train model
    for epoch in range(opt.epochs):
        print("begin to train epochs:{0}",epoch)
        cumulative_loss =0
        prev_loss = 0
        lr_schedule.step()
        for i,data in enumerate(train_loader):
            images,label = data[0][0],data[1][0]
            if int(opt.gpu_id)>=0:
                images = images.cuda()
                label = label.cuda()
            features = model(images)
            loss = loss_criterion(features,label,beta)
            loss.backward()
            optimizer.step()
            optimizer_beta.step()
            cumulative_loss += loss.item()
            if (i+1)%(opt.print_freq)==0:
                print("[Epoch %d,Iter %d] training loss=%f"%(epoch,i+1,cumulative_loss-prev_loss))
                prev_loss = cumulative_loss
            if opt.debug:
                break

        print("[Epoch %d] trainin loss =%f"%(epoch,cumulative_loss))
        # print test val recall index
        names,val_accs = val_model(model,test_loader)
        for name,val_acc in zip(names,val_accs):
            print("Epoch %d,validation:%s=%f"%(epoch,name,val_acc))
    print("job finished")


def val_model(model,test_loader):
    """
    val the model,return the recall@K k=1 index
    :param model: Margin based model to extract feature of 128 dimension
    :param test_loader: Test dataloader to load images data
    :return: the recall@K k=1 index
    """
    model.eval()
    outputs = []
    labels =[]
    with torch.no_grad():
        for data,label in test_loader:
            if int(opt.gpu_id)>=0:
                data = data.cuda()
            feature = model(data)
            outputs += feature.detach().cpu().numpy().tolist()
            labels  += label.numpy().tolist()
        model.train()

    #eval recall@k
    features = np.array(outputs)
    labels = np.array(labels)

    return evaluate_emb(features,labels)

def evaluate_emb(features,labels):
    """
    evaluate embedding in recall
    :param features:
    :param labels:
    :return:
    """
    d_mat = get_distance_matrix(features)
    names =[]
    accs =[]
    for k in [1,2,4,8,16]:
        names.append('Recall@%d'%k)
        correct,cnt = 0.0,0.0
        for i in range(features.shape[0]):
            d_mat[i,i]=1e10
            nns = d_mat[i].argpartition(k)[:k]
            if any(labels[i] ==labels[nn] for nn in nns):
                correct +=1
            cnt +=1
        accs.append(correct/cnt)
    return names,accs # names is a list of ["Recall@K",,,,] accs is a list of [float_value]



def get_distance_matrix(x):
    """
    compute the distance matirx of features,
    :param x: np.ndarray in shape (n,d) d is 128
    :return: distance matrix of [n,n] for distance in each vector
    """
    squrare = np.sum(x*x,axis=1,keepdims=True)
    distance_squrare = squrare + squrare.transpose() -2*np.dot(x,x.transpose())
    return distance_squrare




if __name__=='__main__':
    print("begin to train the model of margin based loss")
    train()
