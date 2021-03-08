import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from PIL import Image
from functools import partial
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import pandas as pd
import re
from collections import defaultdict
import json
import time
import random

tqdm = partial(tqdm, position=0, leave=True)

def load_fname_label(dname):
    data_x_fname = glob.glob(dname + '/Fire/*.jpg')
    data_y = np.ones(len(data_x_fname))
    data_x_fname.extend(glob.glob(dname + '/Neutral/*.jpg'))
    data_y = np.concatenate((data_y,np.zeros(len(data_x_fname)-len(data_y))),axis=None)
    return data_x_fname, data_y

def load_data(file, read_size=(150,150)):  
    img = load_img(file,
                   target_size=read_size,
                   color_mode = "rgb",
                   interpolation="nearest")
    return img_to_array(img)/255

def load_xy(l_fname,size):
    x = []
    y = []
    for name in tqdm(l_fname, desc='creating x,y'):
        x.append(load_data(name,size))
        y.append(is_fire(name))
    x = np.array(x)
    y = np.array(y)
    return x,y

def is_fire(fname):
    if re.search('Fire',fname):
        return 1
    else:
        return 0
    
def augment_data(xt,yt,datagen, size):
    X = []
    Y = []
    pbar = tqdm(total=size,desc='data augmenting:')
    for x_batch, y_batch in datagen.flow(xt, yt):
        for x, y in zip(x_batch,y_batch):
            X.append(x)
            Y.append(y)
        if len(X) > size:
            break
        pbar.update(32)
    pbar.close()
    return np.array(X), np.array(Y)
    
def img_meta(img_files,read_size):
    meta_dict = defaultdict(list)
    for im_file in tqdm(img_files, file=sys.stdout, desc='loading images meata:'):
        _, name = os.path.split(im_file)
        img = img_to_array(load_img(im_file))
        height = img.shape[0]
        width = img.shape[1]
        channel = img.shape[2]
        img_max = np.max(img)
        img_min = np.min(img)
        meta_dict['file_name'].append(name)
        meta_dict['height'].append(height)
        meta_dict['width'].append(width)
        meta_dict['channel'].append(channel)
        meta_dict['hw_ratio'].append(height/width)
        meta_dict['img_size'].append([height, width])
        meta_dict['img_mean'].append(np.mean(img))
        meta_dict['img_median'].append(np.median(img))
        meta_dict['img_std'].append(np.std(img))
        meta_dict['img_min'].append(img_min)
        meta_dict['img_max'].append(img_max)
        meta_dict['img_range'].append(img_max - img_min)
        meta_dict['label'].append(is_fire(im_file))
    return meta_dict

def model_eval(predict,y):
    tp,tn,fp,fn = 0,0,0,0
    for i in range(len(y)):
        if predict[i] == 1:
            if y[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if y[i] == 1:
                fn += 1
            else:
                tn += 1
    print('\tConfusion Matrix')
    print('------------------------------')
    print('\t   Predict')
    print('            "1" |  "0"')
    print('Actual "1"|',tp,'|',fn)
    print('       "0"|',fp,' |',tn)
    print('------------------------------')
    print('')
    
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = (2)*(precision*recall)/(precision+recall)
    print('accuracy = ', accuracy)
    print('precision = ', precision)
    print('recall = ', recall)
    print('F1 = ',f1)
    return tp,fp,fn,tn

def plot_train_log(train_log):   
    train_loss = train_log['loss']
    val_loss = train_log['val_loss']
    plt.figure(figsize=(10,8))
    plt.plot(list(range(len(train_loss))),train_loss)
    plt.plot(list(range(len(val_loss))), val_loss)
    plt.title('loss')

    train_acc = train_log['accuracy']
    val_acc = train_log['val_accuracy']
    plt.figure(figsize=(10,8))
    plt.plot(list(range(len(train_acc))),train_acc)
    plt.plot(list(range(len(val_acc))), val_acc)
    plt.title('accuracy')

fname, label = load_fname_label('./data/Train')
fname_test, tst_label = load_fname_label('./data/Test')
# meta = img_meta(fname)

size = (150,150)
# fname_val = fname[:100]
# fname_val.extend(fname[-100:])
# fname_train = fname[100:-100]
x_t, y_t = load_xy(fname,size)
# x_v, y_v = load_xy(fname_val,size)
x_tst,y_tst = load_xy(fname_test,size)




datagen_train = ImageDataGenerator(rotation_range=90,
                            width_shift_range=0.3,
                            height_shift_range=0.4,
                            horizontal_flip=True,
                            vertical_flip=True,
                            fill_mode='nearest')

# datagen_val = ImageDataGenerator(rotation_range=90,
#                             width_shift_range=0.3,
#                             height_shift_range=0.4,
#                             horizontal_flip=True,
#                             vertical_flip=True,
#                             fill_mode='nearest')
datagen_test = ImageDataGenerator(rotation_range=90,
                            width_shift_range=0.3,
                            height_shift_range=0.4,
                            horizontal_flip=True,
                            vertical_flip=True,
                            fill_mode='nearest')


# X_val = []
# y_val = []

datagen_train.fit(x_t)
# datagen_val.fit(x_v)
datagen_test.fit(x_tst)
# X_train, y_train = augment_data(x_t,y_t,datagen_train,6000)
# X_val, y_val = augment_data(x_v,y_v, datagen_val, 600)
# X_test, y_test = augment_data(x_tst,y_tst, datagen_val, 600)


X_train = x_t
y_train = y_t
X_test = x_tst
y_test = y_tst

filters = [32,64,128]
nodes = [256,512,128]
convs = range(1,6)
denses = range(4)
batch = 32

for c in convs:
    for d in denses:
        nc = []
        nd = []
        

        model = Sequential()
        n1 = random.choice(filters)
        nc.append(str(n1))
        model.add(Conv2D(n1, 3, padding="same", activation="relu", input_shape=(150,150,3)))
        model.add(MaxPool2D(pool_size=(2, 2)))
        
        for i in range(c):
            n1 = random.choice(filters)
            f1 = random.choice([3,3,3,3,5,5])
            st = random.choice([1,2])
            info = f'{n1},{f1},{st}'
            nc.append(str(info))
            model.add(Conv2D(n1, f1,strides=(st, st), padding="same", activation="relu"))
            if i < 3:
                model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())
        for i in range(d):
            n2 = random.choice(nodes)
            nd.append(str(n2))
            model.add(Dense(n2,activation="relu"))
            if i > 1:
                model.add(Dropout(0.2))
                
        model.add(Dense(1, activation="sigmoid"))
        cnn_name = f"cnn-con{c}-{'-'.join(nc)}-dn{d}-{'-'.join(nd)}-{int(time.time())}"
        tb = TensorBoard(log_dir='./logsCloud/'+cnn_name)
        print(cnn_name)

        model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
#         early_stopping = EarlyStopping(monitor='val_loss',
#                               patience=10,
#                               verbose=0,
#                               mode='auto')
        model_save = ModelCheckpoint('./models/'+ cnn_name+'.hdf5',
                             save_best_only=True,
                             monitor='val_loss',
                             mode='min')

#         reduce_lr = ReduceLROnPlateau(monitor='val_loss',
#                               factor=0.5,
#                               patience=5,
#                               verbose=1,
#                               min_delta=5e-5,
#                               mode='min')

#         train_log_cnn = model.fit(X_train,y_train,
#                                   batch_size=128,
#                                   epochs=100,
#                                   validation_data=(X_val, y_val),
#                                  callbacks=[early_stopping, model_save, reduce_lr,tb])
        
        model.fit_generator(generator=datagen_train.flow(X_train,y_train),
                                steps_per_epoch=4000//batch,
                                validation_data=datagen_test.flow(X_test,y_test),
                                validation_steps=len(X_test)//batch,
                                callbacks=[tb,model_save],
                                epochs=10)
