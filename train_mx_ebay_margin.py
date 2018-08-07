# mxnet train ebay dataset,model copy from mxnet example deep learning emebding

import argparse
import logging
import time
import numpy as np

from bottleneck import argpartition
import mxnet as mx
from data import getCUB200,getEbayCrossClassData,getEbayInClassData
from data import getDeepInClassFashion,getDeepCrossClassFashion
import os
from mxnet import gluon
import mxnet.gluon.model_zoo.vision as vision
from mxnet import autograd as ag
from mxnet import nd
from models.mx_margin_model import MarginLoss,MarginNet
from utils import Visulizer
from configs import opt as opt_conf
import ipdb
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='train a model for image classification.')
parser.add_argument('--data', type=str, default='CUB_200_2011',
                    help='path of data.')
parser.add_argument('--embed-dim', type=int, default=128,
                    help='dimensionality of image embedding. default is 128.')
parser.add_argument('--batch-size', type=int, default=70,
                    help='training batch size per device (CPU/GPU). default is 70.')
parser.add_argument('--batch-k', type=int, default=5,
                    help='number of images per class in a batch. default is 5.')
parser.add_argument('--gpus', type=str, default='',
                    help='list of gpus to use, e.g. 0 or 0,2,5. empty means using cpu.')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of training epochs. default is 20.')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='optimizer. default is adam.')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate. default is 0.0001.')
parser.add_argument('--lr-beta', type=float, default=0.1,
                    help='learning rate for the beta in margin based loss. default is 0.1.')
parser.add_argument('--margin', type=float, default=0.2,
                    help='margin for the margin based loss. default is 0.2.')
parser.add_argument('--beta', type=float, default=1.2,
                    help='initial value for beta. default is 1.2.')
parser.add_argument('--nu', type=float, default=0.0,
                    help='regularization parameter for beta. default is 0.0.')
parser.add_argument('--factor', type=float, default=0.5,
                    help='learning rate schedule factor. default is 0.5.')
parser.add_argument('--steps', type=str, default='12,14,16,18',
                    help='epochs to update learning rate. default is 12,14,16,18.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed to use. default=123.')
parser.add_argument('--model', type=str, default='resnet50_v2',
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--save-model-prefix', type=str, default='margin_loss_model',
                    help='prefix of models to be saved.')
parser.add_argument('--use_pretrained', action='store_true',
                    help='enable using pretrained model from gluon.')
parser.add_argument('--kvstore', type=str, default='device',
                    help='kvstore to use for trainer.')
parser.add_argument('--log-interval', type=int, default=20,
                    help='number of batches to wait before logging.')
parser.add_argument('--debug',action='store_true',
                    help='enable debug to run through the model pipline')
parser.add_argument('--use_viz',action='store_true',
                    help='enable using visualization to vis the loss curve')
parser.add_argument('--name',type=str,default='cub200',
                    help='the train instance name')
parser.add_argument('--load_model_path',type=str,default='checkpoints/Fashion_In.params',
                    help='the trained model')

opt = parser.parse_args()
opt.save_model_prefix = opt.name # force save model prefix to name
logging.info(opt)
# Settings.
mx.random.seed(opt.seed)
np.random.seed(opt.seed)

batch_size = opt.batch_size

os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
gpus = [] if opt.gpus is None or opt.gpus is '' else [
    int(gpu) for gpu in opt.gpus.split(',')]
num_gpus = len(gpus)

batch_size *= max(1, num_gpus)
context = [mx.gpu(i) for i in gpus] if num_gpus > 0 else [mx.cpu()]
steps = [int(step) for step in opt.steps.split(',')]

# Construct model.
kwargs = {'ctx': context, 'pretrained': opt.use_pretrained}
net = vision.get_model(opt.model, **kwargs)

if opt.use_pretrained:
    # Use a smaller learning rate for pre-trained convolutional layers.
    for v in net.collect_params().values():
        if 'conv' in v.name:
            setattr(v, 'lr_mult', 0.01)

net.hybridize()
net = MarginNet(net.features, opt.embed_dim, opt.batch_k)
beta = mx.gluon.Parameter('beta', shape=(100000,))
data_dict={'CUB_200_2011':{'data_dir':'CUB_200_2011','func':getCUB200},
           'EbayInClass':{'data_dir':'Stanford_Online_Products','func':getEbayInClassData},
           'EbayCrossClass':{'data_dir':'Stanford_Online_Products','func':getEbayCrossClassData},
           'DeepFashionInClass':{'data_dir':'DeepInShop','func':getDeepInClassFashion},
           'DeepFashionCrossClass':{'data_dir':'DeepInShop','func':getDeepCrossClassFashion}}
if opt.debug:
    ipdb.set_trace()
train_dataloader,val_dataloader = data_dict[opt.data]['func'](os.path.join('data/',data_dict[opt.data]['data_dir']),
                                                              batch_k=opt.batch_k,batch_size=opt.batch_size)
# if opt.data=='Ebay':
#     train_dataloader,val_dataloader = getEbayData(os.path.join('data/',opt.data),batch_k=opt.batch_k,batch_size=batch_size )
# elif opt.data=='CUB_200_2011':
#     train_dataloader,val_dataloader = getCUB200(os.path.join('data/',opt.data),batch_k=opt.batch_k,batch_size=batch_size )
#train_dataloader has datashape [1,batch_size,channel,W,H] for image data,[1,batch_size,1] for label
#test_dataloader has datashape  [batch_size,channel,W,H] for image data,[batch_size,1] for label
# use viz
if opt.use_viz:
    viz = Visulizer(host=opt_conf.vis_host,port=opt_conf.vis_port,env='mx_margin'+opt.name)
    viz.log(str(opt))
    viz.log("start to train mxnet marging model name:%s"%(opt.name))

def get_distance_matrix(x):
    """Get distance matrix given a matrix. Used in testing."""
    if opt.use_viz:
        viz.log("begin to compute distance matrix")
    square = nd.sum(x ** 2.0, axis=1, keepdims=True)
    distance_square = square + square.transpose() - (2.0 * nd.dot(x, x.transpose()))
    return nd.sqrt(distance_square)

def evaluate_emb(emb, labels):
    """Evaluate embeddings based on Recall@k."""
    d_mat = get_distance_matrix(emb)
    #d_mat = d_mat.asnumpy()
    #labels = labels.asnumpy() #directory operate on mxnet.ndarray if convert to numpy,would cause memeory error

    names = []
    accs = []
    for i in range(emb.shape[0]):
        d_mat[i,i]=1e10
    index_mat = nd.argsort(d_mat)
    nd.waitall()
    if opt.use_viz:
        viz.log("nd all dist mat")
    for k in [1, 2, 4, 8, 16]:
        names.append('Recall@%d' % k)
        correct, cnt = 0.0, 0.0
        index_mat_part = index_mat[:,:k]
        for i in range(emb.shape[0]):
            if any(labels[i] == labels[nn] for nn in index_mat_part[i]):
                correct +=1
            cnt +=1
        # for i in range(emb.shape[0]):
        #     d_mat[i, i] = 1e10
        #     nns = argpartition(d_mat[i], k)[:k]
        #     if any(labels[i] == labels[nn] for nn in nns):
        #         correct += 1
        #     cnt += 1
        accs.append(correct/cnt)
    return names, accs


def test(ctx):
    """Test a model."""
    if opt.use_viz:
        viz.log("begin to valid")

    outputs = []
    labels = []
    for i,batch in enumerate(val_dataloader):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        # after split data is list of two data batch
        for x in data:
            outputs.append(net(x)[-1])
        labels +=label
        if (i+1)%(opt.log_interval*2) ==0:
            viz.log("valid iter {0}".format(i))
    outputs = nd.concatenate(outputs, axis=0)
    labels = nd.concatenate(labels, axis=0)
    viz.log("begin to eval embedding search")
    return evaluate_emb(outputs, labels)

def get_lr(lr, epoch, steps, factor):
    """Get learning rate based on schedule."""
    for s in steps:
        if epoch >= s:
            lr *= factor
    return lr

def train(epochs,ctx):
    """Training function."""
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)

    opt_options = {'learning_rate': opt.lr, 'wd': opt.wd}
    if opt.optimizer == 'sgd':
        opt_options['momentum'] = 0.9
    if opt.optimizer == 'adam':
        opt_options['epsilon'] = 1e-7
    trainer = gluon.Trainer(net.collect_params(), opt.optimizer,
                            opt_options,
                            kvstore=opt.kvstore)
    if opt.lr_beta > 0.0:
        # Jointly train class-specific beta.
        # See "sampling matters in deep embedding learning" paper for details.
        beta.initialize(mx.init.Constant(opt.beta), ctx=ctx)
        trainer_beta = gluon.Trainer([beta], 'sgd',
                                     {'learning_rate': opt.lr_beta, 'momentum': 0.9},
                                     kvstore=opt.kvstore)

    loss = MarginLoss(margin=opt.margin, nu=opt.nu)


    best_val =0.0
    for epoch in range(epochs):
        tic = time.time()
        prev_loss,cumulative_loss = 0.0,0.0

        # Learning rage schedule
        trainer.set_learning_rate(get_lr(opt.lr,epoch,steps,opt.factor))
        if opt.use_viz:
            viz.log("Epoch {0} learning rate = {1}".format(epoch,trainer.learning_rate))
        if opt.lr_beta>0:
            trainer_beta.set_learning_rate(get_lr(opt.lr_beta,epoch,steps,opt.factor))
            viz.log("Epoch {0} beta learning rate={1}".format(epoch,trainer_beta.learning_rate))

        #Inner training loop
        for i,batch_data in enumerate(train_dataloader):
            batch = batch_data[0][0] # batch_data is a tuple(x,y) x shape is [1,70,3,227,227]
            label = batch_data[1][0]
            data = gluon.utils.split_and_load(batch,ctx_list=ctx,batch_axis=0)
            label = gluon.utils.split_and_load(label,ctx_list=ctx,batch_axis=0)

            # After split,the data and label datatype is list
            Ls = []
            with ag.record():
                for x, y in zip(data, label):
                    a_indices, anchors, positives, negatives, _ = net(x)

                    if opt.lr_beta > 0.0:
                        L = loss(anchors, positives, negatives, beta, y[a_indices])
                    else:
                        L = loss(anchors, positives, negatives, opt.beta, None)

                    # Store the loss and do backward after we have done forward
                    # on all GPUs for better speed on multiple GPUs.
                    Ls.append(L)
                    cumulative_loss += nd.mean(L).asscalar()

                for L in Ls:
                    L.backward()

            # Update.
            trainer.step(batch.shape[0])
            if opt.lr_beta > 0.0:
                trainer_beta.step(batch.shape[0])

            if (i + 1) % opt.log_interval == 0:
                viz.log('[Epoch {0}, Iter {1}] training loss={2}'.format(
                    epoch, i + 1, cumulative_loss - prev_loss))
                if opt.use_viz:
                    viz.plot('margin_loss',cumulative_loss-prev_loss)
                prev_loss = cumulative_loss
            if opt.debug:
                import ipdb
                ipdb.set_trace()
                break

        viz.log('[Epoch {0}] training loss={1}'.format(epoch, cumulative_loss))
        viz.log('[Epoch {0}] time cost: {1}'.format(epoch, time.time() - tic))

        names, val_accs = test(ctx)
        for name, val_acc in zip(names, val_accs):
            viz.log('[Epoch {0}] validation: {1}={2}'.format(epoch, name, val_acc))
        viz.plot('recall@1',val_accs[0])

        if val_accs[0] > best_val:
            best_val = val_accs[0]
            viz.log('Saving {0}'.format(opt.save_model_prefix))
            net.save_parameters('checkpoints/%s.params' % opt.save_model_prefix)
    return best_val


def extract_feature():
    """
    extract data feature vector and save
    :param model:
    :param dataloader:
    :return:
    """
    global net
    deepfashion_csv = 'checkpoints/deepfashion.csv'
    net.initialize()
    net.collect_params().reset_ctx(context)
    net.load_parameters(opt.load_model_path,ctx=context)
    import csv
    f = open(deepfashion_csv,'w')
    writer = csv.writer(f,dialect='excel')

    for i,batch in tqdm(enumerate(val_dataloader)):
        batch_size = batch[0].shape[0]
        data = gluon.utils.split_and_load(batch[0], ctx_list=context, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=context, batch_axis=0)
        # after split data is list of two data batch
        small_batch_feature = []
        for x in data:
            feature = net.extract(x)
            small_batch_feature.append(feature)
        image_id = np.arange(i*batch_size,(i+1)*batch_size).reshape(-1,1) # prepare the image_id
        vector = nd.concatenate(small_batch_feature,axis=0).asnumpy() # concatenate the feature
        label = np.array([x.asnumpy() for x in label]).reshape(-1,1)
        result = np.hstack((image_id,label,vector))
        writer.writerows(result)
    print("finished extract feature")
    f.close()
    return "True finished"





if __name__ == '__main__':
    import ipdb
    ipdb.set_trace()
    #best_val_recall = train(opt.epochs,context)
    #print("Best validation Recall@1:%.2f"%(best_val_recall))

    result = extract_feature()
    print(result)