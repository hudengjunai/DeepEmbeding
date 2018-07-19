from configs import opt
import visdom
import torch as t
import time
import numpy as np
class Visulizer(object):
    """the object interface to store train trace to website"""
    def __init__(self,host=opt.vis_host,port=opt.vis_port,env=opt.vis_env):
        self.vis = visdom.Visdom(server=host,port=port,env=env)

        self.index ={}
        self.log_text=""

    def reinit(self,env='default'):
        self.vis = visdom.Visdom(server=opt.vis_host,port=opt.vis_port,env=opt.vis_env)
        return self

    def plot(self,name,y):
        """plot loss:1.0"""
        x = self.index.get(name,0)
        self.vis.line(Y=np.array([y]),X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x==0 else 'append')
        self.index[name] = x+1

    def img(self,name,img_,**kwargs):
        """
        :param name: the window name
        :param img_: img shape and data type,t.Tensor(64,64),Tensor(3,64,64),Tensor(100,1,64,64)
        :param kwargs:
        :return:
        """
        self.vis.images(t.Tensor(img_).cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs)

    def log(self,info,win='log_text'):
        """self.log({loss:1,'lr':0.0001}"""
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'), \
            info=info))
        self.vis.text(self.log_text, win)

if __name__=='__main__':
    """nohup python -m visdom.server --port-8088 & 
    this to start visdom server"""
    viz = Visulizer(host='http://192.168.3.13',port=8088,env='street')
    viz.log("this is a start")
    viz.plot('loss',2.3)
    viz.plot('loss',2.2)
    viz.plot('loss',2.1)

    viz.img('origin',np.random.random((10,3,224,224)))