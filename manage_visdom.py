from utils import Visulizer
from configs import opt
from visdom import Visdom
from utils import Visulizer

viz = Visulizer(opt.vis_host,opt.vis_port,env='main')
print(viz)
viz.delete_env('dmldml3')
print("finished")