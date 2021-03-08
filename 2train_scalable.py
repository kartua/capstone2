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
import amornsaensuk as am



def data_gen(fname,batch_size=32):
    random.shuffle(fname)
    while True:
        num_batch = len(fname)//batch_size
        for n in range(num_batch+1):
            x,y = am.load_xy(fname[n*batch_size:(n+1)*batch_size],(150,150),verbose=False)
            yield x,y

if __name__=='__main__':
    fname, label = am.load_fname_label('./data/Train')
    fname_test, tst_label = am.load_fname_label('./data/Test')




    filters = [32,64,128,256]
    convs = [2]
    denses = [2]
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
                nc.append(str(n1))
                model.add(Conv2D(n1, 3, padding="same", activation="relu"))
                if i < 3:
                    model.add(MaxPool2D(pool_size=(2, 2)))

            model.add(Flatten())
            for i in range(d):
                n2 = random.choice(filters)
                nd.append(str(n2))
                model.add(Dense(n2,activation="relu"))
                if i > 1:
                    model.add(Dropout(0.2))

            model.add(Dense(1, activation="sigmoid"))
            cnn_name = f"cnn-con{c}-{'-'.join(nc)}-dn{d}-{'-'.join(nd)}-{int(time.time())}"
            tb = TensorBoard(log_dir='./logsGen/'+cnn_name)
            print(cnn_name)

            model_save = ModelCheckpoint('./models/'+ cnn_name+'.hdf5',
                                 save_best_only=True,
                                 monitor='val_loss',
                                 mode='min')
            
            model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

            model.fit_generator(generator=data_gen(fname,32),
                                steps_per_epoch=len(fname)//batch,
                                validation_data=data_gen(fname_test,batch),
                                validation_steps=len(fname_test)//batch,
                                callbacks=[tb,model_save],
                                epochs=10)
