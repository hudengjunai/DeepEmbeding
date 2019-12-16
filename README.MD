# Deep Embedding Learning for Image Retrieval
---
# Deep Embedding Introduction  
DeepEmbedding 是使用深度学习的方法把多种媒体映射嵌入到相同向量空间，在统一空间中进行搜索的技术。  
本项目通过视觉级别搜索，细粒度类别（实例检索）和图像-文本互搜的方式来测试通用多媒体检索。  
# 图像检索问题的一般解法  
DeepEmbedding旨在使用深度度量学习（DeepMetric)或者深度哈希（DeepHash）的方法学习关系保持的空间映射函数，能将视觉空间映射到低维嵌入空间,使用向量搜索引擎进行搜索.第一个问题为特征提取问题,即为本实验研究的问题,第二个问题为特征搜索问题.第二个问题解决方案可参见ANNS（近似邻近搜索）[NNSearchService](https://github.com/EigenLab/NNSearchService) 。  
## Note
- 本项目实现了基于Multi-npair-loss的度量学习应用于检索和基于Sampling margin loss的方法进行检索
- 具体复现参见论文Triplet loss[FaceNet: ](http://arxiv.org/abs/1503.03832)  
[N-pair loss](http://www.nec-labs.com/uploads/images/Department-Images/MediaAnalytics/papers/nips16_npairmetriclearning.pdf)  
[Margin lossSampling Matters in Deep Emebedding Learning](https://www.cs.utexas.edu/~cywu/projects/sampling_matters/)   
[BatchHard](https://arxiv.org/abs/1703.07737)   

# 实验结果 
(可点击下载百度网盘链接,查看图片)
- 在StanfordOnlineProduct训练，计算NMI聚类指标 nmi=0.866，对验证集的向量嵌入进行T-SNE降维后可以看出，降维图像约 43M，百度网盘链接在下：
- [margin_based loss :DeepFashion](https://pan.baidu.com/s/1zLZX24qBb_Op1vsry4LX6w)https://pan.baidu.com/s/1zLZX24qBb_Op1vsry4LX6w  
计算指标：nmi=0.866  
- [Mc-n-pair loss:StanfordOnlineProduct](https://pan.baidu.com/s/12eNTVsRFu--SYMW8P8HPfQ)https://pan.baidu.com/s/12eNTVsRFu--SYMW8P8HPfQ  
计算指标：nmi=0.830  


# 关于本项目的使用
1.下载相应的数据集
2.采用不同的loss类型对模型进行训练
  run train cub200 model
  ```angular2html
  nohup python train_mx_ebay_margin.py --gpus=1 --batch-k=5 --use_viz --epochs=30 --use_pretrained --steps=12,16,20,24 --name=CUB_200_2011 --save-model-prefix=cub200 > mycub200.out 2>&1 &
  ```
  run train stanford_online_product
  ```angular2html
  nohup python train_mx_ebay_margin.py --batch-k=2 --batch-size=80 --use_pretrained --use_viz --gpus=0 --name=Inclass_ebay --data=EbayInClass --save-model-prefix=ebayinclass > mytraininclass_ebay.log 2>&1
```

  
3.后续工作：
  - 测试R-MAC NetVLAD等网络在视觉检索中的效果和评测
  - 测试使用GAN的方法增强检索效果
  ```angularjs

    __Deep Adversarial Metric Learning__  
    Deep Metric learning cannot get full used the easy negative examples,to in [Deep Adversarial Metirc Learning](http://openaccess.thecvf.com/content_cvpr_2018/papers/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.pdf) proposal a new framework called DAML  
    __DeepMetric and Deep Hashing Apply__  
     apply method to Fashion,vehicle and Person-ReID domain  
    __Construct a datasets__ crawl application-domain data  
```
# Dataset

[CUB200_2011](http://www.vision.caltech.edu/visipedia/CUB-200.html): A small part of ImageNet  
[LFW](http://vis-www.cs.umass.edu/lfw/):face dataset  
[StanfordOnlineProducts](http://cvgl.stanford.edu/projects/lifted_struct/): a lot of types of products(furniture,bicycle,cups)  
[Street2Shop](http://www.tamaraberg.com/street2shop/):products data set from ebay  
[DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html):all are colthes    

# 图像检索的应用
- Face indentification: deep metric learning for face cluster,from [FaceNet](http://arxiv.org/abs/1503.03832) to [SphereFace](http://ieeexplore.ieee.org/document/8100196/)
- Person ReIdentification:deep metric learning for Pedestrian Re-ID,from [MARS](https://pdfs.semanticscholar.org/c038/7e788a52f10bf35d4d50659cfa515d89fbec.pdf) to [NPSM&SVDNet](https://blog.csdn.net/u013982164/article/details/79608100)
- Vehicle Search:deep metric learning for fake-licensed car or Vehicle retrieval.
- Street2Products:search fashion clothe from street photos or in-shop photos,namely visual-search.from [DeepRanking](https://users.eecs.northwestern.edu/~jwa368/pdfs/deep_ranking.pdf) to [DAML](http://openaccess.thecvf.com/content_cvpr_2018/papers/Duan_Deep_Adversarial_Metric_CVPR_2018_paper.pdf)
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


# 部分DeepHash的工作  
DeepHash能将图片直接哈希到汉明码，使用faiss的IVF Binary 系列搜索加速，通过存储量大大减少。

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


## 视觉和文本共同嵌入 Visual-semantic-align embedding(cross modal retrieval)
1.[VSE++: Improving Visual-Semantic Embeddings with Hard Negatives](http://arxiv.org/abs/1707.05612)  
2.[Dual-Path Convolutional Image-Text Embedding with Instance Loss](http://arxiv.org/abs/1711.05535)  
```bash
python train_vse.py --params
```

## 其他类型的搜索方式
1.Sketch based [Deep Sketch Hashing: Fast Free-hand Sketch-Based Image Retrieval](https://github.com/ymcidence/DeepSketchHashing)  
2.Text cross modal based [Deep Cross-Modal Hashing](https://github.com/jiangqy/DCMH-CVPR2017)

## 近似近邻搜索加速
ANNS (Approximation Nearest Neighbor Search) to search a query vector in gallery database.
测试数据集
- [SIFT1M](http://corpus-texmex.irisa.fr/) typical 128-dim sift vector
- [DEEP1B](http://sites.skoltech.ru/compvision/noimi/)，proposed by yandex.inc,this is a deep descriptor
- [GIST1M](http://corpus-texmex.irisa.fr/) typical 512-dim gist vector

### papers
-- PQ based  
1.将传统的标量量化，转成分段乘积量化[Product Quantization for Nearest Neighbor Search](http://ieeexplore.ieee.org/document/5432202/)  
2.类似于Cartisian Quantization，将向量整体进行旋转，使得聚类的分段坐标轴和向量对齐，聚类中心点和数据之间的重建误差小，压缩损失就小[Optimized Product Quantization](http://ieeexplore.ieee.org/document/6678503/)  
3.[Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors](http://arxiv.org/abs/1802.02422)  
4.粗量化使用双倒排，可以降低聚类维度，增加聚类中心点，使用Multi-Sequence算法提高粗略命中速度[The Inverted MultiIndex](http://cache-ash04.cdn.yandex.net/download.yandex.ru/company/cvpr2012.pdf)  
5.多义码，将汉明码距离和量化中心点的距离建立映射关系，对entry-point有过滤作用 [Polysemous codes](https://arxiv.org/abs/1609.01882)  

-- Graph Based  
1.可参见NSW 浏览小世界 论文，结合跳表结构构造的索引[Approximate nearest neighbor algorithm based on navigable small world graphs](https://linkinghub.elsevier.com/retrieve/pii/S0306437913001300)   
2.[Efficient and robust approximate nearest neighbor search using hierarchical Navigable Small World graphs](http://arxiv.org/abs/1603.09320)   
7.[EFANNA : An Extremely Fast Approximate Nearest Neighbor Search Algorithm Based on kNN Graph](http://arxiv.org/abs/1609.07228)  
8.[A Revisit on Deep Hashings for Large-scale Content Based Image Retrieval](http://arxiv.org/abs/1711.06016)   
9.[RobustiQ A Robust ANN Search Method for Billion-scale Similarity Search on GPUs](http://users.monash.edu/~yli/assets/pdf/icmr19-sigconf.pdf)    
10.[GGNN: Graph-based GPU Nearest Neighbor Search](https://arxiv.org/pdf/1912.01059.pdf)   
11.[Zoom: Multi-View Vector Search for Optimizing Accuracy, Latency and Memory](https://www.microsoft.com/en-us/research/uploads/prod/2018/08/zoom-multi-view-tech-report.pdf)    
12. [Vector and Line Quantization for Billion-scale Similarity Search on GPUs](https://arxiv.org/pdf/1901.00275.pdf)    

-- Hamming Code  
1.[Fast Exact Search in Hamming Space with Multi-Index Hashing](http://arxiv.org/abs/1307.2982)  
2.[Fast Nearest Neighbor Search in the Hamming Space](http://link.springer.com/10.1007/978-3-319-27671-7_27)   
4.[Web-Scale Responsive Visual Search at Bing](http://arxiv.org/abs/1802.04914)  
5.[Recurrent Binary Embedding for GPU-Enabled Exhaustive Retrieval from Billion-Scale Semantic Vectors](http://arxiv.org/abs/1802.06466) 
### library
1.[faiss] 当前faiss包含IVF,IMI，PQ,OPQ,PCA,二级残差量化ReRank-PQ,HNSW,Link and Code 等各种类型的索引引擎
2.索引格式选择：高容量，低精度 IMI+OPQ+reRank  
              高精度，选择HNSW,当前NSG索引不支持增量插入，没有采用
3.sptag,基于空间切分建图的方式   
### framework    
1.vearch :by JingDong AI.  和深度学习模型深度贴合     
2.milvus :by zilliz.  以数据库的思路来做的    


