from mxnet.gluon.data import Dataset

from data.mxdata.basic_module.basic_transform import default_transform


class DeepInFashion(Dataset):
    def __init__(self,dir_root,batch_k=4,batch_size=80,is_train = True,transform = default_transform):
        self.root = dir_root
        self.batch_k = batch_k
        self.batch_size = batch_size
        self._transform = transform
        self.is_train = is_train
        self.train_ids = set()
        self.test_ids = set()
        self.boxes = {}
        with open(os.path.join(self.root,'Anno','list_item_inshop.txt.'),'r') as f_instance:
            self.instance_count = int(f_instance.readline().strip())
            #self.instance_ids = list(f_instance.readlines())
            self.images_files = [ [] for _ in range(self.instance_count)]
        with open(os.path.join(self.root,'Anno','list_eval_partition.txt'),'r') as f_parti:
            f_instance.readline() # read pictures number
            f_instance.readline() # read information
            for line in f_instance.readlines():
                path,item_id,status = line.strip().split(' ')
                int_id = int(item_id.split('_')[-1])
                self.images_files[int_id].append(path)
                self.train_ids.add(int_id)
                self.test_ids.add(int_id)
        with open(os.path.join(self.root,'Anno','list_bbox_inshop.txt'),'r') as f_bbox:
            f_bbox.readline() # read count
            f_bbox.readline() # read description
            for line in f_bbox.readlines():
                list_info = line.strip().split(' ')
                path,box = list_info[0],list_info[-4:]
                self.boxes[path]=box
        #read instance ,split set,bbox data

        sub_list_test = self.images_files[list(self.test_ids)]
        self.test_len = 0
        for small_list in sub_list_test:
            self.test_len += len(small_list)
    def build_structure(self):
        """build the folder to id structure dataset"""
        print("the img_root:%s"%(self.root))
        img_root = os.path.join(self.root,'img')
        self.super_types = {}
        for sexual in os.listdir(img_root):
            for clothe_type in os.listdir(os.path.join(self.root,'img',sexual)):
                ids = os.listdir(os.path.join(self.root,'img',sexual,clothe_type))
                self.super_types[sexual+'_'+clothe_type] = [int(instance_id.split('_')[-1]) for instance_id in ids]

    def __len__(self):
        if self.is_train:
            return len(self.train_ids)
        else:
            return self.test_len

    def sampled_batch_data(self):
        """choose an super_types, then choose the batch with batch_k iamges with bbox crop"""

    def __getitem__(self, index):
        if self.is_train:
            fnames,labels = self.sampled_batch_data()
            # read all data from fname and then stack.
        else:





