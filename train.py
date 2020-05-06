import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from layers import densenet
from tensorflow.keras import layers
import h5py
import cv2
import math
import imgaug as ia
import imgaug.augmenters as iaa
from triplet_model.triplet_loss import batch_hard_triplet_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')


DATA_LENGTH,IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL = 2952,300,100, 1
IMG_PER_CLASS=6
CLASSES=123*4
TRAIN_RATE=0.8

'''read_data'''
CACHE_FILE="data/FV_data_cache.h5"
if not os.path.exists(CACHE_FILE):
    print("未发现处理好的数据文件，正在处理...")
    print(data.shape)
    h5f = h5py.File(CACHE_FILE, 'w')
    h5f["X"] = data
    h5f.close()
else:
    h5f = h5py.File(CACHE_FILE, 'r')
    data = h5f["X"][:]
    h5f.close()
    print("发现处理好的数据文件，正在读取...")

data = data.reshape((CLASSES, IMG_PER_CLASS, IMG_HEIGHT, IMG_WIDTH,IMG_CHANNEL ))
labels = np.arange(CLASSES)
permutation = np.random.permutation(CLASSES)
train_data = data[permutation[:int(TRAIN_RATE*CLASSES)]]
test_data = data[permutation[int(TRAIN_RATE*CLASSES):]]
train_labels = labels[permutation[:int(TRAIN_RATE*CLASSES)]]
test_labels = labels[permutation[int(TRAIN_RATE*CLASSES):]]
train_data=train_data/255
test_data=test_data/255

"未改动"
def batch_loader(data, labels, batch_size=64):
    data_classes, img_per_class, h, w, c = data.shape
    data_length = data_classes * img_per_class
    data = data.reshape((data_length, h, w, c))
    permutation = np.random.permutation(data_length)
    for i in range(0, data_length, batch_size):
        batch_permutation = permutation[i: i+batch_size]
        yield data[batch_permutation], labels[batch_permutation//img_per_class]


"改之后的）"
def batch_loader(data, labels, batch_size=64):
    data_classes, img_per_class, h, w, c = data.shape
    data_length = data_classes * img_per_class
    permutation = np.random.permutation(data_classes)
    for i in range(0, data_classes, batch_size):
        batch_permutation = permutation[i: i+batch_size]
        data_ = data[batch_permutation].reshape((batch_size*img_per_class,h, w, c))
        labels_= [val for val in labels[batch_permutation] for i in range(img_per_class)]
        labels_=np.array(labels_)
        yield data_, labels_

'''shujukuozheng'''
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
aug = iaa.Sequential(
    [
        sometimes(iaa.CropAndPad(
            percent=(-0.05, 0.05),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )),
        sometimes(iaa.Affine(
            scale={"x": (0.95, 1.05), "y": (0.95, 1.05)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, # translate by -20 to +20 percent (per axis)
            rotate=(-3, 3), # rotate by -45 to +45 degrees
            shear=(-3, 3), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        sometimes(iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
        ])),
        sometimes(iaa.Add((-10, 10), per_channel=0)),
        sometimes(iaa.Sharpen(alpha=0.1, lightness=1, name=None, deterministic=False, random_state=None)),
        sometimes(iaa.Emboss(alpha=0.3, strength=1, name=None, deterministic=False, random_state=None)),
        sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=5, per_channel=False, name=None, deterministic=False, random_state=None)),
        sometimes(iaa.Dropout(p=0.005, per_channel=False, name=None, deterministic=False, random_state=None)),
        sometimes(iaa.ElasticTransformation(alpha=0.5,
                          sigma=0,
                          name=None,
                          deterministic=False,
                          random_state=None))

    ],
    random_order=True
)

def amplify(X):
    data_length = X.shape[0]
    flip_methods = [lambda _: _]
    method_indices = np.random.choice(len(flip_methods), DATA_LENGTH)
    for idx in range(data_length):
        flip_method = flip_methods[method_indices[idx]]
        X[idx] = flip_method(X[idx])
    return X

'''网络结构'''
num_epoch = 100
batch_size = 64
learning_rate = 1e-4
step_per_epoch = len(train_data) // batch_size
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    learning_rate,
    decay_steps=step_per_epoch*2,
    decay_rate=1,
    staircase=False)

base_model = tf.keras.applications.DenseNet121(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL),
                                               include_top=False,
                                               weights=None)
input = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL))
x = input
x = base_model(x)
x = tf.keras.layers.GlobalAveragePooling2D(name="gad")(x)
# x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128,name="dense")(x)
x = tf.keras.backend.l2_normalize(x, axis=-1)
output = x
model = tf.keras.Model(inputs=input,outputs=output)
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
train_loss_recorder = tf.keras.metrics.Mean(name='train_loss')
test_loss_recorder = tf.keras.metrics.Mean(name='test_loss')



'''training '''
@tf.function
def train_on_batch(labels_batch, data_batch, margin):
    with tf.GradientTape() as tape:
        embeddings = model(tf.cast(data_batch,float), training=True)
        triplet_loss = batch_hard_triplet_loss(labels_batch, embeddings, margin, squared=True)
    grads = tape.gradient(triplet_loss, model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
    train_loss_recorder(triplet_loss)
@tf.function
def test_on_batch(labels_batch, data_batch, margin):
    embeddings = model(tf.cast(data_batch,float))
    test_loss = batch_hard_triplet_loss(labels_batch, embeddings, margin, squared=True)
    test_loss_recorder(test_loss)
#     return test_loss
margin=0.5
for i in range(2000):
    train_loss_recorder.reset_states()
    test_loss_recorder.reset_states()
    # Training
    for j, (data_batch,labels_batch) in enumerate(batch_loader(train_data, train_labels, batch_size=batch_size)):
#         embeddings = model(data_batch.astype(np.float32))
        data_batch=amplify(data_batch)
        train_on_batch(labels_batch, data_batch, margin)
        checkpoint = tf.train.Checkpoint(myAwesomeModel=model)
        manager = tf.train.CheckpointManager(checkpoint, directory='./save', max_to_keep=3)
        if j % 100 == 0:                              # 每隔100个Batch保存一次
            path = manager.save(checkpoint_number=j)        # 保存模型参数到文件
            print("model saved to %s" % path)
        print('[Training] batch {:5d} loss: {:.6f} '.format(j, train_loss_recorder.result()),end='\r')
    for j, (data_batch,labels_batch) in enumerate(batch_loader(test_data, test_labels, batch_size=batch_size)):
        test_on_batch(labels_batch, data_batch, margin)
        print('[test] batch {:5d} loss: {:.6f} '.format(j, test_loss_recorder.result()),end='\r')
    print('epoch {:5d} loss: {:.6f}  test_loss: {:.6f}  '.format(i,
                np.mean(train_loss_recorder.result()),
                np.mean(test_loss_recorder.result())
                ))


'''把测试集保存成csv文件，横轴128向量，数列种类数'''
def save_csv(test_labels,test_data):
    import pandas as pd
    list = np.zeros(shape=(len(test_labels),128))
    index,id= [], []
    for i in range(len(test_labels)):
        index.append(test_labels[i])
        id.append(i)
    index=np.array(index)
    id=np.array(id)
#     print(test_data[0].shape)
    for j in range(len(index)):
        embeddings=model(tf.cast(test_data[j],float))
        embeddings=tf.reduce_mean(embeddings,axis=0)
        for i in range(128):
            list[j][i] = embeddings[i]
    test=pd.DataFrame(columns=np.arange(128),index=index,data=list)
    test.to_csv('test.csv')

save_csv(test_labels,test_data)

'''top 1 准确率'''
def distance(A, B):
    # 欧式距离
    d = tf.reduce_sum(tf.square(A - B), -1)
    return d

def top_1(test_labels,test_data):
    t=0
    import pandas as pd
    for j in range(len(test_labels)):
        for k in range(IMG_PER_CLASS):
            d_=[]
            em1=model(np.expand_dims(test_data[j][k],axis=0))
            data = pd.read_csv("test.csv")
            for i in range(test_data.shape[0]):
                d_.append(distance(data.values[i][1:],em1))
            d_=tf.cast(d_,float)
            d_=tf.reshape(d_, [len(test_labels)])
            d1 = tf.nn.top_k(d_,len(test_labels),sorted=False)
            if tf.equal(d1[-1][-1],j):
                t=t+1
    top_1 = t/(len(test_labels)*IMG_PER_CLASS)
    return top_1

top_1 = top_1(test_labels,test_data)
print(top_1)

'''混淆矩阵'''
matrix = np.zeros(shape=(len(test_labels),IMG_PER_CLASS,128))
in_distance_num = math.ceil((IMG_PER_CLASS*(IMG_PER_CLASS-1))/2)
out_distance_num = IMG_PER_CLASS*IMG_PER_CLASS
for i in range(len(test_labels)):
            matrix[i]=(model(test_data[i]))


'''ROC曲线'''
data = tf.reshape(matrix,(len(test_labels),IMG_PER_CLASS,128))

'''类内距离：用的距离矩阵求取的，128*15（6个中任选两个）'''
in_distance = []
for i in range(data.shape[0]):
    dot_product=tf.cast(tf.matmul(data[i], tf.transpose(data[i])),float)
    square_norm = tf.linalg.diag_part(dot_product)
    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
    distances = tf.reshape(np.triu(distances),(1,IMG_PER_CLASS*IMG_PER_CLASS))
    distances = tf.squeeze(distances,0)
    for i in range(len(distances)):
        if distances[i]>0:
            in_distance.append(distances[i])
in_distance = np.array(in_distance)
'''类间距离：每个样本与其他类间任取3个'''
out_distance = []
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        for k in range (3):
            c1 =np.random.choice(data.shape[0])
            if i != c1:
                t1 =np.random.choice(data.shape[1])
                out_distance.append(tf.reduce_sum(tf.square(data[i][j] - data[c1][t1]), -1))
            else:
                i=i-1
                break
out_distance = np.array(out_distance)
thresld=np.arange(0.01,2.5,0.01)
FRR, FAR = [],[]
eer = 1
j = 2
for k in range(len(thresld)):
    m ,n= 0,0
    for i in range(len(in_distance)):
        if in_distance[i]>thresld[k]:
            m=m+1
    frr=m/len(in_distance)
    FRR.append(frr)
    for i in range(len(out_distance)):
        if out_distance[i]<thresld[k]:
            n=n+1
    far=n/len(out_distance)
#         print(far)
    FAR.append(far)
    if (abs(frr-far)<0.01):
        eer = abs(frr+far)/2
#         print(thresld[k])
plt.plot(FAR,FRR,'-',label='FRR')
# plt.plot(thresld,FAR,'+-',label='FAR')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.0,1.0),loc=1,borderaxespad=0)
plt.show()
print ('EER is: ',eer)
