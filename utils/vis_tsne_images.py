import pandas as pd
import numpy as np
from PIL import Image
from lapjv import lapjv
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import matplotlib as mlp
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
def load_img(file_list,in_dir):
    #pred_img = [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))]
    pred_img =file_list
    img_collection = []
    for idx, img in enumerate(pred_img):
        img = os.path.join(in_dir, img)
        img_collection.append(Image.open(img))
    return img_collection

def save_tsne_grid(img_list, x2d, out_res,crop_size,in_dir):
    """
    plot all the images in X_2d pictures
    :param img_collection: the image
    :param X_2d: the point
    :param out_res: the output picture resolution
    :return:
    """
    out_img = np.ones((out_res+crop_size,out_res+crop_size,3),dtype='uint8')
    out_img = out_img*255

    i=0
    for img_path,point in tqdm(zip(img_list,x2d)):
        i +=1
        point = point*out_res
        px = int(point[0])
        py = int(point[1])
        img = Image.open(os.path.join(in_dir,img_path))
        img.thumbnail((crop_size,crop_size))

        a = np.array(img)

        try:
            h,w = a.shape[:2]
            if len(a.shape)==3:
                out_img[py:py + h, px:px + w]= a
        except Exception as e:
            print(e)
            print(a.shape)
            print(img_path)
        # if i%5000==4999:
        #     tm = out_img.astype('uint8')
        #     tm_pl_img = Image.fromarray(tm)
        #     tm_pl_img.save('checkpoints/tsne_product_{0}.jpg'.format(i+1))

    out_img = out_img.astype('uint8')
    pl_img = Image.fromarray(out_img)
    pl_img.save('checkpoints/tsne_product.jpg')


def generate_tsne(activations):
    perplexity=30
    tsne = TSNE(perplexity=perplexity, n_components=2, init='random')
    X_2d = tsne.fit_transform(activations) # activations dtype is numpy.ndarray
    X_2d -= X_2d.min(axis=0)
    X_2d /= X_2d.max(axis=0)
    return X_2d

def visualize(im_files_list,features=None,tsne_size=20000,crop_size=100):
    """
    visualize t-sne data
    :param im_files_list: image file list
    :param features: image features numpy.ndarray shape (n,512)
    :return:
    """
    print("dimension deduction from 512 features ...")
    feature_2d = generate_tsne(features)
    #feature_2d = np.load('x2d.npy')
    print("build t-sne image ... ...")
    save_tsne_grid(im_files_list, feature_2d, tsne_size,crop_size,'data/Stanford_Online_Products')


def nmi(gt_class,features):
    """
    normal mutual information,for features
    :param im_class: np.ndarray,  shape [n,1],dtype=np.int32
    :param features: image features to clustering ,numpy.ndarray [n,512]
    :return:
    """

    gt_class = gt_class - min(gt_class)
    n_cluster = len(set(gt_class))
    model = KMeans(n_clusters=n_cluster)
    Y=model.fit(features)
    cl_class = Y.labels_
    score = normalized_mutual_info_score(gt_class,cl_class)
    print("the normal_mutal_info_score",score)




if __name__=='__main__':
    """
    read compute data and visualize t-sne picture,then comput nmi index 
    """
    features_file = 'checkpoints/online_product_compute.csv'
    test_info_file =  'data/Stanford_Online_Products/Ebay_test.txt'

    vectors = None
    features = pd.read_csv(features_file,header=None)
    id_class = features.iloc[:,0:2]
    id_class = np.array(id_class)
    vectors = np.array(features.iloc[:,2:])

    image_id_path= pd.read_table(test_info_file, header=0, delim_whitespace=True)
    file_list = np.array(image_id_path.path)

    #visualize(file_list,vectors)
    file_class = np.array(image_id_path.class_id)
    file_class = file_class.astype(np.int32)
    nmi(file_class,vectors)



