from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from PIL import Image
from functools import partial
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from collections import defaultdict
import random
import glob

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

def load_xy(l_fname,size,verbose=True):
    x = []
    y = []
    if verbose == True:
        for name in tqdm(l_fname, desc='creating x,y'):
            x.append(load_data(name,size))
            y.append(is_fire(name))
    else:
        for name in l_fname:
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
    if (tp+fp) == 0:
        precision = 0
    else:
        precision = tp/(tp+fp)
        
    if (tp+fn) == 0:
        recall = 0
    else:    
        recall = tp/(tp+fn)
    if (precision+recall) == 0:
        f1 = 0
    else:    
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
    plt.plot(list(range(len(train_loss))),train_loss,label='train')
    plt.plot(list(range(len(val_loss))), val_loss,label='val')
    plt.title('loss')
    plt.legend()

    train_acc = train_log['accuracy']
    val_acc = train_log['val_accuracy']
    plt.figure(figsize=(10,8))
    plt.plot(list(range(len(train_acc))),train_acc,label='train')
    plt.plot(list(range(len(val_acc))), val_acc,label='val')
    plt.title('accuracy')
    plt.legend()
