## Deep Metric Learning for Image retrieval
Deep Metric Learning for image retrieval:

note:
this project implement the multi-class N-pair loss,Margin-base loss and HDC

the origin paper is :

Triplet loss[FaceNet: A Unified Embedding for Face Recognition and Clustering](http://arxiv.org/abs/1503.03832)

N-pair loss[Improved Deep Metric Learning with Multi-class N-pair Loss Objective](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf)

Margin loss[Sampling Matters in Deep Emebedding Learning](https://www.cs.utexas.edu/~cywu/projects/sampling_matters/)

L1 Triplet Batch_hard softmargin [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737)

# Dataset

[CUB200_2011](http://www.vision.caltech.edu/visipedia/CUB-200.html): A small part of ImageNet

[LFW](http://vis-www.cs.umass.edu/lfw/):face dataset

[StanfordOnlineProducts](http://cvgl.stanford.edu/projects/lifted_struct/): a lot of types of products(furniture,bicycle,cups)

[Street2Shop](http://www.tamaraberg.com/street2shop/):products data set from ebay

[DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html):all are colthes


# Deep embeding usage:
#### Face indentification: deep metric learning from FaceNet to Angular-Face

#### Person ReIdentification:

#### Vehicle Search:

#### Street2Products

# About This project
1.download data store in data directory
2.train model with different loss type and dataset
3.run with mxnet-server or pytorch-flask-server


