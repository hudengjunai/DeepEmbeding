import torch
import torch.optim as optim
from data import EbayDataset
import os
from configs import opt
from models import ModGoogLeNet,NpairLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import Visulizer
import csv
import numpy as np

def train(**kwargs):
    print("run train")
    opt.parse(kwargs)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

    if opt.debug:
        import ipdb
        ipdb.set_trace()
    model =ModGoogLeNet(embeding_size=opt.embeding_size)
    if opt.dml_model_path:
        model.load(opt.dml_model_path)
    if opt.use_gpu:
        model = model.cuda()
    #model.freeze_model(level=opt.freeze_level)

    if opt.use_viz:
        viz = Visulizer(host=opt.vis_host,port=opt.vis_port,env='dml'+opt.vis_env)
        viz.log("start to train dml npair mc model")

    #loss function
    criterion = NpairLoss(l2_reg=opt.l2_reg)
    lr = opt.lr
    m = opt.momentum
    optimizer = optim.SGD([{'params':model.level1_2.parameters()},
                           {'params': model.level_3_4.parameters()},
                           {'params': model.level_5_6.parameters()},
                           {'params': model.level_7.parameters()},
                           {'params':model.fc.parameters(),'lr':10*lr}],lr=lr,momentum=m)
    #optimizer = optim.SGD(model.parameters(),lr=lr,momentum=m)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,1,0.2)

    # data and dataloader
    train_data = EbayDataset(dir_root=opt.ebay_dir, train=True, batch_size=opt.batch_size)
    cycle_length = len(train_data)
    #val_data = EbayDataset(dir_root=opt.ebay_dir, train=False, batch_size=opt.batch_size)
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=opt.num_workers)
    #val_dataloader = DataLoader(val_data, batch_size=60, shuffle=False, num_workers=opt.num_workers)

    print("dataloader setted ,begin to train")

    #f = open('dml_log.out','w')
    for epoch in range(opt.max_epoch):
        lr_scheduler.step()
        train_loss = 0

        for i,data in enumerate(train_dataloader):
            # if i in [200, 800, 1500]:
            #     lr_scheduler.step()

            data = data[0]
            if opt.use_gpu:
                data = data.cuda()
            optimizer.zero_grad()
            feature = model(data)
            batch_size = data.size(0)
            target = torch.arange(0, int(batch_size / 2), dtype=torch.int64).cuda()
            loss  = criterion(feature,target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            freq = int(opt.print_freq)
            if i%freq==(freq-1):
                average_loss = train_loss /opt.print_freq
                #f.write("iteration:{0},dml_loss:{1}\n".format(i+ epoch*cycle_length,average_loss))
                #f.flush()
                if opt.use_viz:
                    viz.plot('dml_loss',average_loss)
                train_loss =0
            if opt.debug:
                break
        #f.write("epoch:{0} finished,begin to valid test".format(epoch))
        model.save()
        # if epoch>1 and epoch%5==0:
        #     val(model,val_dataloader,epoch)
        if opt.debug:
            #f.write("finish one iter")
            break
    #f.write("finish train epoch {0}".format(opt.max_epoch))
    #f.close()


def val(model,dataloder,epoch):
    """
    this val model will calculate the nmi index.normal mutual information
    :param model: the emebding model
    :param dataloder: val dataloder
    :return:
    """
    # prepare file model to extract feature
    file_name = 'checkpoints/online_product_{0}.csv'.format(epoch)
    f = open(file_name,'w')
    writer = csv.writer(f,dialect='excel')
    model.eval()
    # feature extreat,fisrt for all image,image_id,class_id extract the feature vector
    for i,(data,image_id,class_id) in enumerate(dataloder):
        if opt.use_gpu:
            data = data.cuda()
        feature = model(data) # the feature is [batch,512] vector
        vector = feature.cpu().detach().numpy() if opt.use_gpu else feature.numpy()
        image_id = image_id.numpy().reshape(-1,1)
        class_id = class_id.numpy().reshape(-1,1)
        result = np.hstack(image_id,class_id,vector)

        #write the data to dataframe file
        writer.writerows(result)
        if opt.debug:
            print("test one batch of val data and save to csv file")
            break
    f.close()
    # clustering to centriod, second, for all image_id,and corresponding feature vector,using kmeans cluster to fixed |class_id|
    #featuredata = pd.read_csv(file_name,header=None)

    # for distribution from origin and cluster distribution.compute the nmi by sklearn metric nmi


    model.train()
    print("finished cluster and evalution")


def compute(**kwargs):
    print("run compute_vector")
    opt.parse(kwargs)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

    if opt.debug:
        import ipdb
        ipdb.set_trace()
    model =ModGoogLeNet(embeding_size=opt.embeding_size)
    if opt.dml_model_path:
        model.load(opt.dml_model_path)
    if opt.use_gpu:
        model = model.cuda()

    val_data = EbayDataset(dir_root=opt.ebay_dir, train=False, batch_size=opt.batch_size)
    val_dataloader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    file_name = 'checkpoints/online_product_compute.csv'
    f = open(file_name, 'w')
    writer = csv.writer(f, dialect='excel')
    model.eval()
    # feature extreat,fisrt for all image,image_id,class_id extract the feature vector
    for i, (data, image_id, class_id) in enumerate(val_dataloader):
        if opt.use_gpu:
            data = data.cuda()
        feature = model(data)  # the feature is [batch,512] vector
        vector = feature.cpu().detach().numpy() if opt.use_gpu else feature.numpy()
        image_id = image_id.numpy().reshape(-1, 1)
        class_id = class_id.numpy().reshape(-1, 1)
        result = np.hstack([image_id, class_id, vector])

        # write the data to dataframe file
        writer.writerows(result)
        if opt.debug:
            print("test one batch of val data and save to csv file")
            break
    f.close()


def help():
    """print function use information"""
    print("""this file help to train product train:
    exanple --:
          python train_mc_npair.py help
          python train_mc_npair.py train --gpu_id=3 --debug=True
          python train_mc_npair.py train --gpu_id=2 --batch_size=72
          python train_mc_npair.py train --gpu_id=3 --lr=0.0003 --batch_size=72
          python train_mc_npair.py train --gpu_id=0 --debug=True --dml_model_path=checkpoints/DMLGoogle_0710_20\:24\:04.pth
          python train_mc_npair.py train --batch_size=120 --gpu_id=3 --lr=0.0001  --debug=True --dml_model_path=checkpoints/DMLGoogle_0710_20\:24\:04.pth
          python train_mc_npair.py compute --batch_size=300 --gpu_id=2 --dml_model_path=checkpoints/DMLGoogle_0714_07:51:44.pth --num_workers=6
""")

if __name__=='__main__':
    import fire
    fire.Fire()