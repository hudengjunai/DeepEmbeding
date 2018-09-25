# this is a A/B test from image_nn_prod and image_metric_taobao128

old_index = 'image_nn_prod'
new_index = 'image_metric_taobao128'

from .copy_nn import get_net()
from .copy_nn import get_target_colection
from .copy_nn import get_nn_config
from .copy_nn import get_db

if __name__=='__main__':
    dev = get_db()
    host,path = get_nn_config()
    net = get_net(0)
    nn_128 = get_target_colection(db)

