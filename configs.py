import warnings
class DefaultConfig(object):
    num_classes = 13 # street2shop classes
    train_classify_dir = "/data/jh/notebooks/hudengjun/VisNet/classify"
    valid_classify_dir = "/data/jh/notebooks/hudengjun/VisNet/"
    persist = 'data/persist.csv'
    lr=0.001
    lr_step = 50
    use_gpu = True
    gpu_id = 2
    load_model_path=None
    num_workers = 4
    momentum=0.89
    max_epoch = 800
    print_freq = 40
    batch_size = 32
    vis_host="http://hpc3.yud.io"
    vis_port=8088
    vis_env='Street2shop'
    debug = False


    #tripplet dataset config
    ebay_dir = '/data/jh/notebooks/hudengjun/DML/deep_metric_learning/lib/online_products/Stanford_Online_Products/'
    n_pair_train = 'Ebay_train.txt'
    n_pair_test = 'Ebay_test.txt'
    embeding_size=512
    dml_model_path=None
    l2_reg=3e-3
    use_viz=True
    freeze_level=0


def parse(self,kwargs):
    """update dict with kwargs params"""
    for k,v in kwargs.items():
        if not hasattr(self,k):
            warnings.warn("does not has attribute",k)
        setattr(self,k,v)
    print("use config:")
    for k,v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k,':',getattr(self,k))
DefaultConfig.parse = parse
opt = DefaultConfig()
