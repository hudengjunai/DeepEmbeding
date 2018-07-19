import torch
import torch.optim as optim
from data import Street2shop
import os
from configs import opt
from models import VggClassify
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import Visulizer

def val(model,dataloader):
    """run model with data,"""
    model.eval()
    num_total =0
    num_correct =0
    for i,(data,label) in tqdm(enumerate(dataloader)):
        if opt.use_gpu:
            data = data.cuda()
            label = label.cuda()
        score = model(data)
        num_total += data.size(0)
        pred = torch.argmax(score,dim=1)
        acc = torch.eq(pred, label)
        num_correct += acc.sum().item()
        if opt.debug:
            break
    print("valid, correct rate",1.0*num_correct/num_total)
    model.train()

def train(**kwargs):
    opt.parse(kwargs)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)
    #data
    train_data = Street2shop(opt.train_classify_dir,train=True,persist=opt.persist)
    val_data = Street2shop(opt.train_classify_dir,train=False,persist=opt.persist)

    #model
    model = VggClassify(num_classes=opt.num_classes)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model = model.cuda()

    #data loader
    train_dataloader= DataLoader(train_data,batch_size=opt.batch_size,shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,batch_size=opt.batch_size,shuffle=True,num_workers=opt.num_workers)

    #visulizer
    viz = Visulizer(host=opt.vis_host,port=opt.vis_port,env=opt.vis_env)
    viz.log("start to train")
    #loss function
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    m = opt.momentum
    optimizer = optim.SGD(model.parameters(),lr=lr,momentum=m)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,20,0.1)
    for epoch in range(opt.max_epoch):
        lr_scheduler.step()
        train_loss = 0
        for i,(data,label) in tqdm(enumerate(train_dataloader)):
            if opt.use_gpu:
                data = data.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            score = model(data)
            loss = criterion(score,label)
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item()
            if i%opt.print_freq == opt.print_freq-1:
                average_loss = train_loss/opt.batch_size
                viz.plot('loss',average_loss)
                train_loss =0
            if opt.debug:
                break
        print("epoch :{0} finished,begin to valid test".format(epoch))
        model.save()
        val(model,val_dataloader)
        if opt.debug:
            print("finished one iter")
            break
def help():
    """print information"""
    print("""
    useage: python file.py <function> --args=value
    function := train help
    example:
            python {0} train
            python {0} help""")

if __name__=='__main__':
    import fire
    fire.Fire()

