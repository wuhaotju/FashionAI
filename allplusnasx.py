
import os
import cv2
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import *
from keras.applications import inception_resnet_v2
from keras.applications.inception_v3 import preprocess_input
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto(allow_soft_placement=True)
#最多占gpu资源的70%
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
#开始不会给tensorflow全部gpu资源 而是按需增加
#config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
config = tf.ConfigProto()
df_train = pd.read_csv('../train.csv', header=None)
df_train.columns = ['image_id', 'class', 'label']
df_train.head()
classes = ['collar_design_labels', 'neckline_design_labels', 'skirt_length_labels', 
           'sleeve_length_labels', 'neck_design_labels', 'coat_length_labels', 'lapel_design_labels', 
           'pant_length_labels']
len(classes)
df_load = df_train.copy()
df_load.reset_index(inplace=True)
del df_load['index']
#print('{0}: {1}'.format(cur_class, len(df_load)))
df_load.head()
label_counts=['coat_length','collar_design','lapel_design','neck_design','neckline_design','pant_length','skirt_length','sleeve_length']
label_count={'coat_length':8,
             'collar_design':5,
             'lapel_design':5,
             'neck_design':5,
             'neckline_design':10,
             'pant_length':6,
             'skirt_length':6,
             'sleeve_length':9}
label_n={'coat_length_labels':0,
                'collar_design_labels':1,
                 'lapel_design_labels':2,
                 'neck_design_labels':3,
                'neckline_design_labels':4,
                'pant_length_labels':5,
                'skirt_length_labels':6,
                'sleeve_length_labels':7}
n=len(df_load)
y=[np.zeros((n,label_count[x])) for x in label_counts]
count=0
#for i in label_count.keys():
 #   label_n[i+'_labels']=count
  #  count=count+1
n_class = len(df_load['label'][0])
#width = 150
width = 331
#X = np.zeros((n, width, width, 3), dtype=np.uint8)
l=list()
l=list(range(n))
random.shuffle(l)
l_train, l_valid = train_test_split(l, test_size=0.1,random_state=999)
y_train=list()
y_valid=list()
#X_train=np.zeros((len(l_train), width, width, 3), dtype=np.uint8)
X_valid=np.zeros((len(l_valid), width, width, 3), dtype=np.uint8)
v=list(range(n))
random.shuffle(v)
path=list()
for i in tqdm(range(n)):
    ii=v[i]
    tmp_label = df_load['label'][ii]
    tclass=df_load['class'][ii]
    path.append('../train1/{0}'.format(df_load['image_id'][ii]))
    #X[ii] = cv2.resize(cv2.imread('../train1/{0}'.format(df_load['image_id'][i])), (width, width))
    y[label_n[tclass]][i][tmp_label.find('y')] = 1
x_train=list(np.array(path)[l_train])
x_valid1=list(np.array(path)[l_valid])
for i in tqdm(range(len(x_valid1))):
    X_valid[i] = cv2.resize(cv2.imread(x_valid1[i]), (width, width))  
for i in range(len(y)):
    print(i)
    y_train.append(y[i][l_train])
    y_valid.append(y[i][l_valid])
#X_train=X[l_train]
#X_valid=X[l_valid]
###########################################################################
datagen = ImageDataGenerator(
			rotation_range=10,
			width_shift_range=0.2,
			height_shift_range=0.05,
			shear_range=0.1,
			zoom_range=0.05,         
			horizontal_flip=True,
			vertical_flip=False,
			fill_mode='nearest',
			channel_shift_range=10.  
			)
def random_crop(x, random_crop_size, sync_seed=None):
            f=0
            np.random.seed(sync_seed)
            w, h = x.shape[0], x.shape[1]
            rangew = (w - random_crop_size[0]) // 2
            rangeh = (h - random_crop_size[1]) // 2
            offsetw = 0 if rangew == 0 else np.random.randint(rangew)
            offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
            #zero=x[offsetw:offsetw + random_crop_size[0], offseth:offseth + random_crop_size[1], :]==0
            #asum=x[offsetw:offsetw + random_crop_size[0], offseth:offseth + random_crop_size[1], :]>-100
           # print('#####################',w,h,offsetw,offseth)
            return x[offsetw:offsetw + random_crop_size[0], offseth:offseth + random_crop_size[1], :]
def my_datagen(path,y0,size=16,width = 299):
    data=path
    v=list(range(len(data)))
    np.random.shuffle(v)	
    i = 0
    while True:
        X,Yn,Y = [],[],[]
        while len(X)<size:
                img=cv2.resize(cv2.imread(path[v[i]]), (331,331))
                #img=random_crop(img,[299,299])
                img = img[np.newaxis, :]
                x = datagen.flow(img,batch_size=1).next()
                x=np.asarray(x,dtype='uint8')
                X.append(x[0])
                Yn.append(v[i])
                i =i+ 1
                if i >= len(data):
                    i=i-len(data)
                    np.random.shuffle(v)
        for ai in range(len(y0)):
                    Y.append(y0[ai][Yn])
        yield np.array(X),Y
def acc(y_true, y_pred):
    index = tf.reduce_any(y_true > 0.5, axis=-1)
    res = tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))
    index = tf.cast(index, tf.float32)
    res = tf.cast(res, tf.float32)
    return tf.reduce_sum(res * index) / (tf.reduce_sum(index) + 1e-7)
###########################################################################
'''i=0
for x,y in my_datagen(x_train,y_train,size=8):
    i=i+1
    if i>0:
        break'''
    
from keras.callbacks import LambdaCallback
config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config=config))
base_model =NASNetLarge(weights='imagenet', input_shape=(width, width, 3), include_top=False, pooling='avg')
ai=0

input_tensor = Input((width, width, 3))
x = input_tensor
x = Lambda(nasnet.preprocess_input)(x)
x = base_model(x)
x = Dropout(0.6)(x)
x = [Dense(label_count[name], activation='softmax', name=name)(x) for name in label_counts]
model = Model(input_tensor, x)

lr = 1e-6
size =8

#print '------------------------------------------ '
#print ' train lr = ',str(lr)
adam = Adam(lr=lr, beta_2=0.9999, epsilon=1e-08)
'''datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images'''

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal
    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
'''datagen.fit(X_train)'''
X=[]
#batch_size=32
#model.load_weights('../models/all512x.best.h5')
model.load_weights('../models/nasx.h5')
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

checkpointer = ModelCheckpoint(filepath='../models/allnas.best.h5', verbose=1, monitor='val_loss',mode='min',
                               save_best_only=True)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=[acc])

global storedLoss
storedLoss=100000000000000000000
def myCallback(epoch,logs):
    global storedLoss
    #do your comparisons here using the "logs" var.
    print(logs)
    if (logs['val_loss'] < storedLoss):
        storedLoss = logs['val_loss']
        model.save_weights('../models/nasx1.h5')
        '''for i in range(len(model.layers)):
            WandB = model.layers[i].get_weights()
            if len (WandB) > 0: #necessary because some layers have no weights
                np.save("../models/W" + "-" + str(i), WandB[0],False) 
                np.save("../models/B" + "-" + str(i), WandB[1],False)'''

#checkpointer = ModelCheckpoint(filepath='../models/all512x.best.h5', verbose=1,monitor='loss', 
 #                              save_best_only=True)
h=model.fit_generator(
		my_datagen(x_train,y_train,size=size),
       # steps_per_epoch=1,
		steps_per_epoch=len(l_train)//size, 
		validation_data=(X_valid,y_valid),
		#validation_steps=len(l_valid)/size,
		epochs=1,
		callbacks=[EarlyStopping(patience=3),  LambdaCallback(on_epoch_end=myCallback)],
		workers=-1,
		verbose=1)
 #             callbacks=[EarlyStopping(patience=3), checkpointer], 
  #            shuffle=True, 
   #           validation_split=0.1)

#model.evaluate(X_train, y_train, batch_size=128)
#model.evaluate(X_valid, y_valid, batch_size=128)
'''plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.legend(['loss', 'val_acc'])
plt.ylabel('acc')
plt.xlabel('epoch')'''
df_test = pd.read_csv('../test_b.csv', header=None)
df_test.columns = ['image_id', 'class', 'x']
del df_test['x']
df_test.head()
df_load = df_test.copy()
df_load.reset_index(inplace=True)
del df_load['index']


df_load.head()
n = len(df_load)
X_test = np.zeros((n, width, width, 3), dtype=np.uint8)

for i in tqdm(range(n)):
    X_test[i] = cv2.resize(cv2.imread('../test_b2/{0}'.format(df_load['image_id'][i])), (width, width))
test_np = model.predict(X_test, batch_size=64)
result = []

for i, row in df_load.iterrows():
    tmp_list = test_np[label_n[row[1]]][i]
    tmp_result = ''
    for tmp_ret in tmp_list:
        tmp_result += '{:.4f};'.format(tmp_ret)
        
    result.append(tmp_result[:-1])

df_load['result'] = result
df_load.head()
df_load.to_csv('../result/nasx.csv', header=None, index=False)




    
