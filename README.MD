# Deep Embedding Learning
---
# Deep Embedding Introduction
---
DeepEmbedding 是使用深度学习的方法把多种媒体映射嵌入到相同向量空间，在统一空间中进行搜索的技术。
# About This Project  
---
DeepEmbedding旨在使用深度度量学习（DeepMetric)或者深度哈希（DeepHash）的方法学习关系高保持的空间映射函数，能将视觉空间映射到低维嵌入空间，主要解决上述深度度量学习的第一个问题。
第二个问题解决方案可参见ANNS（近似邻近搜索）[NNSearchService](https://github.com/EigenLab/NNSearchService) 
典型的DeepEmbedding有两种思路(根据提取特征为float值densevector或者汉明码)： 
1.使用DeepMetric+VQ的方法,先提取densevector（使用FaceNet等）,如果搜索精度要求高，则使用GPU进行精确距离计算并排序，这是一种精确邻近搜索方法，
比如人脸比对，另一种是近似邻近搜索，使用VQ向量量化的方法，在损失部分精度情况下，加速搜索过程，也即ANNS，这种方法简称Deep-Metric+VQ.  
2.直接使用DeepHash的方法，编码成汉明码，直接进行距离计算。相对于DeepMetric+VQ的非监督量化编码过程，
DeepHash力求使用监督学习的方法得到量化编码（汉明码），减少在向量量化步骤的信息损失，达到更好的特征表达能力。
3.异同比较：DeepMetric和DeepHash都面向图像检索，相同点是：目的相同，学习映射函数；基础方法相似：基于深度学习的方法，目标函数类似：使用了triplet-loss等典型目标函数。
不同点是：输出向量不同：前者是dense-vector,后者是汉明码；精度不同：前者可以提供高精度，能面向reID领域（FaceReID，PersonReID)，后者面向近似搜索。存储容量不同：前者如果不使用ANNS压缩加速，很难扩大容量。后者的compact编码占用空间小，计算速度快。

---  
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
- Face indentification: deep metric learning for face cluster,from [FaceNet](http://arxiv.org/abs/1503.03832) to [SphereFace](http://ieeexplore.ieee.org/document/8100196/)

- Person ReIdentification:deep metric learning for Pedestrian Re-ID,from [MARS](https://pdfs.semanticscholar.org/c038/7e788a52f10bf35d4d50659cfa515d89fbec.pdf) to [NPSM&SVDNet](https://blog.csdn.net/u013982164/article/details/79608100)

- Vehicle Search:deep metric learning for fake-licensed car or Vehicle retrieval.

- Street2Products:search fashion clothe from street photos or in-shop photos,namely visual-search.from [DeepRanking](https://users.eecs.northwestern.edu/~jwa368/pdfs/deep_ranking.pdf) to [DAML](http://openaccess.thecvf.com/content_cvpr_2018/papers/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.pdf)

# About DeepMetric project
1.download data store in data directory
2.train model with different loss type and dataset
  run train cub200 model
  ```angular2html
  nohup python train_mx_ebay_margin.py --gpus=1 --batch-k=5 --use_viz --epochs=30 --use_pretrained --steps=12,16,20,24 --name=CUB_200_2011 --save-model-prefix=cub200 > mycub200.out 2>&1 &
  ```
  run train stanford_online_product
  ```angular2html
  nohup python train_mx_ebay_margin.py --batch-k=2 --batch-size=80 --use_pretrained --use_viz --gpus=0 --name=Inclass_ebay --data=EbayInClass --save-model-prefix=ebayinclass > mytraininclass_ebay.log 2>&1
```

  
3.run with mxnet-server or pytorch-flask-server

4.train result in t-sne and metric
[margin_based loss :DeepFashion](https://pan.baidu.com/s/1zLZX24qBb_Op1vsry4LX6w)
计算指标：nmi=0.866
[Mc-n-pair loss:StanfordOnlineProduct](https://pan.baidu.com/s/12eNTVsRFu--SYMW8P8HPfQ)
计算指标：nmi=0.830
## Future work
__Deep Adversarial Metric Learning__  
Deep Metric learning cannot get full used the easy negative examples,to in [Deep Adversarial Metirc Learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.pdf) proposal a new framework called DAML  
__DeepMetric and Deep Hashing Apply__  
 apply method to Fashion,vehicle and Person-ReID domain  
__Construct a datasets__ crawl application-domain data  

## Deep Metric Learning mile-stone paper:  
1.[DrLIM:Dimensionality Reduction by Learning an Invariant Mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)  
2.[DeepRanking:Learning fine-graied Image Similarity with DeepRanking](https://users.eecs.northwestern.edu/~jwa368/pdfs/deep_ranking.pdf)  
3.[DeepID2:Deep Learning Face Representation by Joint Identification-Verification](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)  
4.[FaceNet:FaceNet: A Unified Embedding for Face Recognition and Clustering](http://arxiv.org/abs/1503.03832)  
5.[Defense:In Defense of the Triplet Loss for Person Re-Identification](http://arxiv.org/abs/1703.07737)  
6.[N-pair:Improved Deep Metric Learning with Multi-class N-pair Loss Objective](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf)  
7.[Sampling:Sampling Matters in Deep Embedding Learning](https://arxiv.org/abs/1706.07567)  
8.[DAML:Deep Adversarial Metric Learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.pdf)  
9.[SphereFace:Deep Hypersphere Embedding for Face Recognition](http://ieeexplore.ieee.org/document/8100196/)


# About the DeepHash project 

## ReImplementation of HashNet
```angular2html
python train_hash.py --params
```


## Deep Hash Learning mile-stone paper:
1.[CNNH:Supervised Hashing for Image Retrieval via Image Representation Learning](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8137/8861)  
2.[DNNH:Simultaneous feature learning and hash coding with deep neural networks](http://ieeexplore.ieee.org/document/7298947/)  
3.[DLBHC:Deep Learning of Binary Hash Codes for Fast Image Retrieval](http://www.iis.sinica.edu.tw/~kevinlin311.tw/cvprw15.pdf)  
4.[DSH:Deep Supervised Hashing for Fast Image Retrieval](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_Deep_Supervised_Hashing_CVPR_2016_paper.pdf)  
5.[SUBIC:SuBiC: A Supervised, Structured Binary Code for Image Search](http://ieeexplore.ieee.org/document/8237358/)  
6.[HashNet:HashNet:Deep Learning to Hash by Continuous](https://arxiv.org/abs/1702.00758)  
7.[DCH:Deep Cauchy Hashing for Hamming Space Retrieval](http://openaccess.thecvf.com/content_cvpr_2018/html/Cao_Deep_Cauchy_Hashing_CVPR_2018_paper.html)  


## Others retrieval method
1.Sketch based [Deep Sketch Hashing: Fast Free-hand Sketch-Based Image Retrieval](https://github.com/ymcidence/DeepSketchHashing)  
2.Text cross modal based [Deep Cross-Modal Hashing](https://github.com/jiangqy/DCMH-CVPR2017)

## ANNS
ANNS (Approximation Nearest Neighbor Search) to search a query vector in gallery database.
#### databse
- [SIFT1M](http://corpus-texmex.irisa.fr/) typical 128-dim sift vector
- [DEEP1B](http://sites.skoltech.ru/compvision/noimi/)，proposed by yandex.inc,this is a deep descriptor
- [GIST1M](http://corpus-texmex.irisa.fr/) typical 512-dim gist vector

### papers

### library


### others
