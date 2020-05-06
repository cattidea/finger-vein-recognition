import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from layers import densenet
from tensorflow.keras import layers
import h5py
import cv2
path = 'data/FV-USM/1st_session/extractedvein'
DATA_LENGTH,IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL=123*4*6,300,100,1
data = np.zeros(shape=(DATA_LENGTH,IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL))
idx = 0
for kind in os.listdir(path):
    kind_path = os.path.join(path,kind)
    for  per_img in os.listdir(kind_path):
        if os.path.splitext(per_img)[-1]==".jpg":
            per_img_path = os.path.join(kind_path,per_img)
            img = cv2.imread(per_img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=-1)
            # print(img.shape,idx)
            data[idx] = img
            idx = idx+1

CACHE_FILE_DATA="NN_net/FV_data_cache.h5"
if not os.path.exists(CACHE_FILE_DATA):
    print("未发现处理好的数据文件，正在处理...")
    # data, labels=get_data()
    h5f = h5py.File(CACHE_FILE_DATA, 'w')
    h5f["X"] = data
    h5f.close()
else:
    h5f = h5py.File(CACHE_FILE_DATA, 'r')
    data = h5f["X"][:]
    h5f.close()
    print("发现处理好的数据文件，正在读取...")
    print(data.shape)

# img = cv2.imread("data/FV-USM/1st_session/extractedvein/vein123_4/01.jpg")
# print(img.shape)
