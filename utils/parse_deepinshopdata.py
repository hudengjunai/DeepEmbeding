import matplotlib.pyplot as plt
import numpy as np
from mxnet.image import imread
import os
bboxfile = r'../Logs/Anno/list_bbox_inshop.txt'
line = None
skip=40009
with open(bboxfile,'r') as f_box:
    f_box.readline() #
    f_box.readline() #
    for i in range(skip):
        f_box.readline()
    line = f_box.readline()

img_dir = r'C:\download\In-shop-clothes'
line_list  = line.strip().split(' ')
path,bbox = line_list[0],line_list[-4:]
print('path:',path,"bbox",bbox)
fig = plt.figure()
plt.subplot(2,1,1)
image = imread(os.path.join(img_dir,path))
bbox=[int(x) for x in bbox]
plt.imshow(image.asnumpy())
plt.subplot(2,1,2)
plt.imshow(image[bbox[1]:bbox[3],bbox[0]:bbox[2]].asnumpy())
plt.show()

img_path= r'C:\Users\Dengjun\Pictures\a.jpg'
img = imread(img_path)
print(img.shape)