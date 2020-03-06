# import pymongo
# import mxnet
# from mxnet import nd
#
# #every time yield 20 items and read iobytes extract feature then insert to new nnindex
#
# import asyncio
# import aiohttp
# from io import BytesIO
# import time
# import requests
#
#
# @asyncio.coroutine
# def get_image(img_url):
#     resp = yield from requests.get(img_url)
#     return resp.content
#
# def save_image(img,fobj):
#     fobj.write(img)
#
# @asyncio.coroutine
# def download_one(img_url,fobj):
#     image = yield from get_image(img_url)
#     save_image(image,fobj)

# !/usr/bin/env python
# import asyncio
# import aiohttp
#
# async def fetch_img(session, url):
#     with aiohttp.Timeout(10):
#         async with session.get(url) as response:
#             assert response.status == 200
#             return await response.read()
#
# loop = asyncio.get_event_loop()
# with aiohttp.ClientSession(loop=loop) as session:
#     img = loop.run_until_complete(
#         fetch_img(session, 'https://cdn.aidigger.com/images/instagram/f95f00da22a2e143e6e457b10544a120.jpeg'))
#     with open("img.png", "wb") as f:
#         f.write(img)

# if __name__ == '__main__':
#     url_list = ['https://cdn.aidigger.com/images/instagram/e2452f9daaad3ef7070adb22ee70958a.jpeg',
#                 'https://cdn.aidigger.com/images/instagram/bd717eaa4c351b842a497e8907b69855.jpeg',
#                 'https://cdn.aidigger.com/images/instagram/189a2af5d9661500b32271ca9b1865be.jpeg',
#                 'https://cdn.aidigger.com/images/instagram/6e70c94dd3fac214c5d7e6c061df2b2f.jpeg',
#                 'https://cdn.aidigger.com/images/instagram/f95f00da22a2e143e6e457b10544a120.jpeg']
#     fobj_list =[BytesIO() for _ in range(len(url_list))]
#     start = time.time()
#     loop = asyncio.get_event_loop()
#     to_do_tasks = [download_one(url,f) for url,f in zip(url_list,fobj_list)]
#     res,= loop.run_until_complete(asyncio.wait(to_do_tasks))
#     print(len(res))
#     print(time.time()-start)


import asyncio
import logging
from contextlib import closing
import aiohttp # $ pip install aiohttp
from io import BytesIO
from PIL import Image
import numpy as np
from pymongo import MongoClient
from mxnet import nd
import mxnet as mx
import mxnet.gluon.data.vision.transforms as T
import mxnet.gluon.model_zoo.vision as vision_model
from models import MarginNet
import mxnet
from mxnet.image import imread

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(message)s')
import requests
import json
import binascii
import numpy as np
from pymongo import MongoClient
from requests import ReadTimeout
from pprint import pprint




#image transform
normalize=T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
test_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    normalize
])

# define mongodb connect
def get_db():
    mongdb={}
    mongdb['host']='cc.com'
    mongdb['port']=3717
    client=MongoClient(host=mongdb['host'],port=mongdb['port'])
    dev=client.get_database('dev')
    dev.authenticate(name='cc',password='cc')
    return dev


@asyncio.coroutine
def download(url, session, semaphore, chunk_size=1<<15):
    with (yield from semaphore): # limit number of concurrent downloads
        file = BytesIO()
        logging.info('downloading %s', file)
        response = yield from session.get(url)
        with closing(response):
            while True: # save file
                chunk = yield from response.content.read(chunk_size)
                if not chunk:
                    break
                file.write(chunk)
        logging.info('done %s', file)
    return file, (response.status, tuple(response.headers.items()))

def get_net(gpu_id):
    param_path = 'checkpoints/Fashion_In.params'
    base_net = vision_model.get_model('resnet50_v2')
    net = MarginNet(base_net.features, 128, batch_k=5)
    context = [mxnet.gpu(gpu_id)]
    net.initialize()
    net.collect_params().reset_ctx(context)
    net.load_parameters(filename=param_path, ctx=context[0])
    return net,context

def get_cursor(db,collection_name,batch_size):
    #define source nn_prod data fetch
    nn_prod = db.get_collection(collection_name)
    cursor = nn_prod.find({},{'vector':0,},batch_size=batch_size)
    return cursor

def get_target_colection(db):
    colletion_name = 'image_metric_taobao128'
    target_collection = db.get_collection(colletion_name)
    return target_collection


def convert_vector_to_ascii(vector):
    """convert a numpy array or a list to bytes, and to make it can be dumped by json, we convert the bytes to string
    """
    if isinstance(vector, (list, np.ndarray, np.generic)):
        vector = np.asarray(vector, dtype=np.float32)
    else:
        raise ValueError("vector must be list or numpy array")
    # add decode to convert base64 bytes to string
    return binascii.b2a_base64(vector.tobytes()).decode()

def get_nn_config(model_name ='image_metric_taobao128'):

    host = 'https://alpha-nnsearch.aidigger.com/api/v1/'
    path = 'model/'+model_name+'/'
    return host,path

# begin to set basic paramter
batch_size=20
urls= []
records = []
db = get_db()
cursor = get_cursor(db,'image_nn_prod',batch_size)
net,context = get_net(0)
host,path = get_nn_config('image_metric_taobao128')
# set basic parameter finished

targe_collection = get_target_colection(db)

loop = asyncio.get_event_loop()
session = aiohttp.ClientSession()
semaphore = asyncio.Semaphore(20)

for item in cursor:
    if len(urls)==batch_size:
        #process
        #with closing(asyncio.get_event_loop()) as loop, closing(aiohttp.ClientSession()) as session:
        try:
            download_tasks = (download(url, session, semaphore) for url in urls)
            result = loop.run_until_complete(asyncio.gather(*download_tasks))
        except Exception as e:
            print(e)
            urls = []
            records = []
            continue

        nd_img_list = []
        succeed_ids = []
        docs = []
        for i,(f_ret,rec) in enumerate(zip(result,records)):
            try:
                pil_img = Image.open(f_ret[0])
                nd_img_list.append(test_transform(nd.array(np.asarray(pil_img))))
                new_rec = {}
                new_rec['_id'] = rec['_id']
                new_rec['_int_id'] = rec['int_id']
                new_rec.update(rec['_source'])
                docs.append(new_rec)
            except Exception as e:
                print(urls[i])
                print(e)


        #nd_img_list = [test_transform(nd.array(np.asarray(Image.open(f_ret[0])))) for f_ret in result ]
        if len(nd_img_list)!=len(records) or len(nd_img_list)< 2:
            if len(nd_img_list)<2:
                print(urls[0])
                print("caution,failed to download all pictures")
                print(result[0][1][0],result[0][1][1])

            records.clear()
            urls.clear()
            docs.clear()
            for f_ret in result:
                try:
                    if not f_ret[0].closed:
                        f_ret[0].close()
                except Exception as e:
                    print(e)
            continue

        nd_tensor_img = nd.stack(*nd_img_list,axis=0)
        nd_tensor_img = nd_tensor_img.as_in_context(context[0])
        data = net.extract(nd_tensor_img)
        data = data.asnumpy()



        doc_types =['image']*len(records)
        vectors = [convert_vector_to_ascii(v) for v in data ]

        ret = requests.post(host + path + "add/batch", json={"docs": docs, "doc_types": doc_types, "vectors": vectors})
        print(ret.json())

        #for annother loop
        doc_types=[]
        vectors =[]
        doc_types=[]
        records = []
        urls=[]
        for f_ret in result:
            try:
                if not f_ret[0].closed:
                    f_ret[0].close()
            except Exception as e:
                print(e)
    else:
        records.append(item)
        urls.append(item['_source']['cdn_url'])



