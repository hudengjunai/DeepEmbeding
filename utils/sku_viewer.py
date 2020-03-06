import pymongo
from pymongo import MongoClient
from pprint import pprint
from pymongo import MongoClient
mongdb={}
mongdb['host']=''
mongdb['port']=3717
client=MongoClient(host=mongdb['host'],port=mongdb['port'])
dev=client.get_database('dev')
dev.authenticate(name='',password='')
print(dev.collection_names())

tao_bao_collection = dev.get_collection('image_faiss_dual_taobao')
print(tao_bao_collection.count())

item = tao_bao_collection.find_one()
pprint(item['_source'])
