import torch
import torch.optim as optim
import argparse
from data import get_hash_dataloader
from models import HashNetRes50,HashLoss
from utils import Visulizer
import torch.nn as nn
import os
import numpy as np
from pprint import pprint
args = argparse.ArgumentParser()
args.add_argument('--gpus',type=str,default='0',help="gpus ids")
args.add_argument('--dataset',type=str,default='coco',help='the dataset name in coco,nus_wide imagent')
args.add_argument('--hash_bit',type=int,default=48,help='the hash bit of deephashing output')
args.add_argument('--iter_nums',type=int,default=10000,help='the max train iter')
args.add_argument('--train_batch',type=int,default=32,help='the train batch_size')
args.add_argument('--lr',type=float,default=0.0001,help='the train learning rate')
args.add_argument('--class_num',type=float,default=1.0,help='the imbalance ratio')
args.add_argument('--viz_env',type=str,default='cocohash',help='the visdom env name')
args.add_argument('--log_interval',type=int,default=20,help='the loss print log interval')
args.add_argument('--snapshot_interval',type=int,default=3000,help='the snapshot archive model interval')
args.add_argument('--test_interval',type=int,default=500,help='the test hash search interval')


def test_model(model,test_loader,database_loader,viz):
    def code_predict(net,loader):
        code = []
        label = []
        for data in loader:
            x,y = data
            if torch.cuda.is_available():
                x = x.cuda()
            x = model(x)
            code.append(x.cpu())
            label.append(y)
        code = torch.cat(code,dim=0)
        code = torch.sign(code) # the quantization sign function
        label = torch.cat(label,dim=0)
        return code.numpy(),label.numpy()
    test_code,test_label = code_predict(model,test_loader)
    database_code,database_label = code_predict(model,database_loader)

    #compute the mean average precision--namely map
    query_num = test_code.shape[0]
    sim = np.dot(database_code, test_code.T)
    ids = np.argsort(-sim, axis=0)
    APx = []

    for i in range(query_num):
        label = test_label[i, :]
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_label[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)
        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
    mAP = np.mean(np.array(APx))
    viz.plot("mAP",str(mAP))






if __name__=='__main__':
    config={}
    ags = args.parse_args()
    config['gpus']=int(ags.gpus)
    os.environ['CUDA_VISIBLE_DEVICES']=ags.gpus
    config['dataset']=ags.dataset
    config['hash_bit'] = ags.hash_bit
    config['iter_nums']= ags.iter_nums
    config['train_batch'] = ags.train_batch
    config['lr']=ags.lr
    config['log_interval'] = ags.log_interval
    config['snapshot_interval'] = ags.snapshot_interval
    config['test_interval'] = ags.test_interval
    config['viz_env'] = ags.viz_env



    #program setting
    config['weight_decay']=0.0005

    config["optimiz_params"] = {"lr": config['lr'], "momentum": 0.9, "weight_decay": 0.0005, "nesterov": True}
    config['lr_scheduler']={"gamma":0.5, "step":2000}
    config["loss"] = {"l_weight": 1.0, "q_weight": 0,
                      "l_threshold": 15.0, "sigmoid_param": 10. / config["hash_bit"],
                      "class_num": ags.class_num}

    pprint(config) # print the config data
    #prepare model and dataset
    model = HashNetRes50(n_bit=config['hash_bit'])
    criteria = HashLoss(hash_bit=config['hash_bit'])

    train1,train2,test_loader,database_loader = get_hash_dataloader(config['dataset'],config['train_batch'],
                                                      config['train_batch']//2,config['train_batch']//2)
    if torch.cuda.is_available():
        model = model.cuda()
    params_list = [{"params":model.feature_layers.parameters(),'lr':1},
              {"params":model.hash_layer.parameters(),'lr':10}]
    optimizer = optim.SGD(params_list,lr=config['lr'],momentum=0.9,weight_decay=config['weight_decay'],nesterov=True)
    lr_schedualer = optim.lr_scheduler.StepLR(optimizer,step_size=2000,
                                              gamma=0.5,last_epoch=-1)

    viz = Visulizer(host='http://hpc3.yud.io',port=8088,env=config['viz_env'])
    viz.log("start the hash learning")
    viz.log(config)
    len_train = len(train1)
    train_loss = 0
    for it in range(config['iter_nums']):

        lr_schedualer.step()
        if it % len_train==0:
            iter1 = iter(train1)
            iter2 = iter(train2)
        train_part1 = iter1.next()
        train_part2 = iter2.next() # same train data two different shuffle

        x1,y1 = train_part1
        x2,y2 = train_part2
        if torch.cuda.is_available():
            x1 = x1.cuda()
            x2 = x2.cuda()
            y1 = y1.cuda()
            y2 = y2.cuda()
        inputs = torch.cat((x1,x2),dim=0)
        labels = torch.cat((y1,y2),dim=0)
        outputs = model(inputs)
        loss = criteria(outputs,labels,sigmoid_param=config["loss"]["sigmoid_param"], \
                             l_threshold=config["loss"]["l_threshold"], \
                             class_num=config["loss"]["class_num"])
        loss.backward()
        train_loss += loss.item()
        if (it+1)%config['log_interval']==0:
            print("Iter: {:05d}, loss: {:.3f}".format(it,train_loss/config['log_interval']))
            train_loss =0
        optimizer.step()

        if it%config['snapshot_interval'] ==0:
            torch.save(nn.Sequential(model),
                       './checkpoints/resnet_{0}_{1}_{2}.pth.tar'.format(config['dataset'],config['hash_bit'],it))
        if it%config['test_interval']==0:
            test_model(model,test_loader,database_loader) # to validate the efficiency of hash code
    viz.log("finish train model")

