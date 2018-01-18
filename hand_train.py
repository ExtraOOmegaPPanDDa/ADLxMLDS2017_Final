#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:41:19 2018

@author: HSIN
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 00:01:14 2018

@author: HSIN
"""


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import sys
import os
import json
import scipy
import numpy as np
import re
import pickle
import time

import imageio
from skimage import img_as_float
from skimage.draw import line,circle
from skimage.io import imread, imsave, imshow

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, BatchNormalization, Permute
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D 
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.optimizers import Adam, RMSprop
from keras.layers import Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications import inception_resnet_v2

import judger_hand
import cv2


np.random.seed(46)

data_path = sys.argv[1]

synth01_sets = ['s000','s001','s002','s003','s004']
synth02_sets = ['s005','s006','s007','s008','s009']
vive_sets = ['air','book']

selection_sets = sys.argv[2].split('*')

restore_exist_path = sys.argv[3]
restore_bbx_path = sys.argv[4]

restore_exist = True
if restore_exist_path == 'None':
    restore_exist = False

restore_bbx = True
if restore_bbx_path == 'None':
    restore_bbx = False


save_exist_path = sys.argv[5]
save_bbx_path = sys.argv[6]


# for large synth data, 
# only the data with its identity_number % synth_sample_D == synth_sample_R
# will be taken in to the training

synth_sample_D = 8
synth_sample_R = 0


vive_h = 460
vive_w = 612

synth_h = 240
synth_w = 320

img_h = 224
img_w = 224


drop_out_ratio = 0.1

exist_vive_epochs = 25
exist_vive_patience = 10

bbx_vive_epochs = 100
bbx_vive_patience = 25

sample_size = 10000

batch_size = 8

sleep_time = 3


exist_lr = 0.0001
bbx_lr = 0.001

Train_Exist = True
Train_BBX = True

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def img_prepro(image_data):
#    retval, image_data = cv2.threshold(image_data,110, 255, cv2.cv2.THRESH_TOZERO)
    image_data = scipy.misc.imresize(image_data, [img_h, img_w, 3])
#    image_data = rgb2gray(image_data)
    return image_data
    


def draw_rectangle(img, x0, y0, x1, y1, color=(0, 0, 0)):
    draw_line(img, x0, y0, x0, y1, color)
    draw_line(img, x0, y0, x1, y0, color)
    draw_line(img, x1, y0, x1, y1, color)
    draw_line(img, x0, y1, x1, y1, color)


def draw_line(img, x0, y0, x1, y1, color=(0, 0, 0)):
    x0 = min(img.shape[1]-1, max(0, x0))
    y0 = min(img.shape[0]-1, max(0, y0))
    x1 = min(img.shape[1]-1, max(0, x1))
    y1 = min(img.shape[0]-1, max(0, y1))
    yy, xx = line(y0,x0,y1,x1)
    img[yy, xx, :] = color



def draw_bbox(img_path, label_path, output_path):
    img = img_as_float(imread(img_path))
    json_item = json.load(open(label_path))
    colors = {'L':(1,0,0),'R':(0,1,0)}
    for htype,box in json_item['bbox'].items():
        draw_rectangle(img, box[0], box[1], box[2], box[3], color=colors[htype])
        draw_rectangle(img, box[0] + 1, box[1] + 1, box[2] + 1, box[3] + 1, color=(0, 0, 0))
    imsave(output_path, img)



X_dict = {}
y_exist_dict = {}
y_bbx_dict = {}

for selection_set in selection_sets:
    
    img_path = 'data/' + selection_set + '/img/'
    label_path = 'data/' + selection_set + '/label/'
    
    if selection_set in synth01_sets:
        img_path = os.path.join('DeepQ-Synth-Hand-01', img_path)
        label_path = os.path.join('DeepQ-Synth-Hand-01', label_path)
    elif selection_set in synth02_sets:
        img_path = os.path.join('DeepQ-Synth-Hand-02', img_path)
        label_path = os.path.join('DeepQ-Synth-Hand-02', label_path)
    elif selection_set in vive_sets:
        img_path = os.path.join('DeepQ-Vivepaper', img_path)
        label_path = os.path.join('DeepQ-Vivepaper', label_path)
    
    the_imgs = sorted(os.listdir(os.path.join(data_path, img_path)))
    the_labs = sorted(os.listdir(os.path.join(data_path, label_path)))
    
    the_imgs = the_imgs[:sample_size]
    the_labs = the_labs[:sample_size]
    
    
    for i in range(len(the_imgs)):
        
        img = the_imgs[i]
        
        if int(img[-9:].replace('.png','')) % synth_sample_D != synth_sample_R:
            if selection_set not in vive_sets:
                continue
        
        if not img.startswith('img'):
            continue
        
        if i % 100 == 0:
            print(selection_set, img)
            
        
        image_path = os.path.join(data_path, img_path, img)
        
        image_data = imageio.imread(image_path)
        image_data = img_prepro(image_data)
        
        X_index = re.sub("\D", "", img)
        X_index = X_index + '_' + selection_set
        X_dict[X_index] = image_data
        
        
        
        image_data_flip = np.fliplr(image_data)
        X_index_flip = X_index + '_flip'
        X_dict[X_index_flip] = image_data_flip
    
    
    
    for i in range(len(the_labs)):
        
        check_bound = True
        
        lab = the_labs[i]
        
        if not lab.startswith('label'):
            continue
        
        if i % 100 == 0:
            print(selection_set, lab)
            
        lab_path = os.path.join(data_path, label_path, lab)
        
        y_bbx_data = []
        y_exist_data = []
        
        y_bbx_data_flip = []
        y_exist_data_flip = []
        
        
        with open(lab_path, 'r') as f:
            
            lab_data = json.load(f)
            
            
            if selection_set in synth01_sets:
                scale_h = synth_h
                scale_w = synth_w
                
            elif selection_set in synth02_sets:
                scale_h = synth_h
                scale_w = synth_w
                
            elif selection_set in vive_sets:
                scale_h = vive_h 
                scale_w = vive_w
            
            if 'L' in lab_data['bbox']:
                
                y_exist_data.append(1)
                
                
                for n in range(4):
                    if lab_data['bbox']['L'][n] < 0:
                        check_bound = False
                    elif n%2 == 0 and lab_data['bbox']['L'][n] > scale_w:
                        check_bound = False
                    elif n%2 == 1 and lab_data['bbox']['L'][n] > scale_h:
                        check_bound = False
                
                y_bbx_data.append( min(max(lab_data['bbox']['L'][0] * img_w / scale_w, 0), img_w) ) # left
                y_bbx_data.append( min(max(lab_data['bbox']['L'][1] * img_h / scale_h, 0), img_h) ) # up
                y_bbx_data.append( min(max(lab_data['bbox']['L'][2] * img_w / scale_w, 0), img_w) ) # right
                y_bbx_data.append( min(max(lab_data['bbox']['L'][3] * img_h / scale_h, 0), img_h) ) # bottom
                
                
                y_bbx_data_flip.append( min(max((scale_w - lab_data['bbox']['L'][2]) * img_w / scale_w, 0), img_w) ) # left
                y_bbx_data_flip.append( min(max((lab_data['bbox']['L'][1]) * img_h / scale_h, 0), img_h) ) # up
                y_bbx_data_flip.append( min(max((scale_w - lab_data['bbox']['L'][0]) * img_w / scale_w, 0), img_w) ) # right
                y_bbx_data_flip.append( min(max((lab_data['bbox']['L'][3]) * img_h / scale_h, 0), img_h) ) # bottom
                
            else:
                
                y_exist_data.append(0)
                
                y_bbx_data.append(-100 * img_w / scale_w)
                y_bbx_data.append(-100 * img_h / scale_h)
                y_bbx_data.append(-100 * img_w / scale_w)
                y_bbx_data.append(-100 * img_h / scale_h)
                
                y_bbx_data_flip.append(-100 * img_w / scale_w)
                y_bbx_data_flip.append(-100 * img_h / scale_h)
                y_bbx_data_flip.append(-100 * img_w / scale_w)
                y_bbx_data_flip.append(-100 * img_h / scale_h)
            
            
            
            if 'R' in lab_data['bbox']:
                
                y_exist_data.append(1)
                
                
                for n in range(4):
                    if lab_data['bbox']['R'][n] < 0:
                        check_bound = False
                    elif n%2 == 0 and lab_data['bbox']['R'][n] > scale_w:
                        check_bound = False
                    elif n%2 == 1 and lab_data['bbox']['R'][n] > scale_h:
                        check_bound = False
                    
                    
                
                
                y_bbx_data.append( min(max(lab_data['bbox']['R'][0] * img_w / scale_w, 0), img_w) ) # left
                y_bbx_data.append( min(max(lab_data['bbox']['R'][1] * img_h / scale_h, 0), img_h) ) # up
                y_bbx_data.append( min(max(lab_data['bbox']['R'][2] * img_w / scale_w, 0), img_w) ) # right
                y_bbx_data.append( min(max(lab_data['bbox']['R'][3] * img_h / scale_h, 0), img_h) ) # bottom
                
                
                y_bbx_data_flip.append( min(max((scale_w - lab_data['bbox']['R'][2]) * img_w / scale_w, 0), img_w) ) # left
                y_bbx_data_flip.append( min(max((lab_data['bbox']['R'][1]) * img_h / scale_h, 0), img_h) ) # up
                y_bbx_data_flip.append( min(max((scale_w - lab_data['bbox']['R'][0]) * img_w / scale_w, 0), img_w) ) # right
                y_bbx_data_flip.append( min(max((lab_data['bbox']['R'][3]) * img_h / scale_h, 0), img_h) ) # bottom
                
            else:
                
                y_exist_data.append(0)
                
                y_bbx_data.append(-100 * img_w / scale_w)
                y_bbx_data.append(-100 * img_h / scale_h)
                y_bbx_data.append(-100 * img_w / scale_w)
                y_bbx_data.append(-100 * img_h / scale_h)
                
                y_bbx_data_flip.append(-100 * img_w / scale_w)
                y_bbx_data_flip.append(-100 * img_h / scale_h)
                y_bbx_data_flip.append(-100 * img_w / scale_w)
                y_bbx_data_flip.append(-100 * img_h / scale_h)
        
        
        y_bbx_data_flip_done = []
        y_bbx_data_flip_done.append(y_bbx_data_flip[4])
        y_bbx_data_flip_done.append(y_bbx_data_flip[5])
        y_bbx_data_flip_done.append(y_bbx_data_flip[6])
        y_bbx_data_flip_done.append(y_bbx_data_flip[7])
        y_bbx_data_flip_done.append(y_bbx_data_flip[0])
        y_bbx_data_flip_done.append(y_bbx_data_flip[1])
        y_bbx_data_flip_done.append(y_bbx_data_flip[2])
        y_bbx_data_flip_done.append(y_bbx_data_flip[3])
        
        y_exist_data_flip_done = []
        y_exist_data_flip_done.append(y_exist_data[1])
        y_exist_data_flip_done.append(y_exist_data[0])
        
        if check_bound:
            y_index = re.sub("\D", "", lab)
            y_index = y_index + '_' + selection_set
            
            y_bbx_dict[y_index] = y_bbx_data
            y_exist_dict[y_index] = y_exist_data
            
            y_index_flip = y_index + '_flip'
            y_bbx_dict[y_index_flip] = y_bbx_data_flip_done
            y_exist_dict[y_index_flip] = y_exist_data_flip_done
            

X = []
y_exist = []
y_bbx = []
index_record = []
    
for idx in list(X_dict):
    try:
        y_exist.append(y_exist_dict[idx])
        y_bbx.append(y_bbx_dict[idx])
        X.append(X_dict[idx])
        index_record.append(idx)
    except:
        continue
        
 


    

X = np.asarray(X)
y_exist = np.asarray(y_exist)
y_bbx = np.asarray(y_bbx)
index_record = np.asarray(index_record)


print('X Shape', X.shape)
print('y_exist Shape', y_exist.shape)
print('y_bbx Shape', y_bbx.shape)

input_dim = (X.shape[-3], X.shape[-2], X.shape[-1])
output_dim_exist = 2
output_dim_bbx = 8


train_valid_ratio = 0.9
indices = np.random.permutation(X.shape[0])
train_idx, valid_idx = indices[:int(X.shape[0] * train_valid_ratio)], indices[int(X.shape[0] * train_valid_ratio):]

train_X, valid_X = X[train_idx,:], X[valid_idx,:]
train_y_exist, valid_y_exist = y_exist[train_idx,:], y_exist[valid_idx,:]
train_y_bbx, valid_y_bbx = y_bbx[train_idx,:], y_bbx[valid_idx,:]

train_index_record, valid_index_record = index_record[train_idx], index_record[valid_idx]


def build_exist_model():

#    cnn = resnet50.ResNet50(weights='imagenet', include_top=True)
    
#    cnn = inception_v3.InceptionV3(weights='imagenet',
#                    input_shape = (img_h, img_w, 3),
#                    include_top=False,
#                    pooling='avg')
    
    cnn = inception_resnet_v2.InceptionResNetV2(weights='imagenet',
                    input_shape = (img_h, img_w, 3),
                    include_top=False,
                    pooling='avg')
    
    
    
    for layer in cnn.layers:
        layer.trainable = True
        
        
    cnn.trainable = True
    x = cnn.output
    x = Dropout(drop_out_ratio)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(drop_out_ratio)(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    Output = Dense(2, activation='sigmoid')(x)
    model = Model(cnn.input, Output)
    
    return model



def build_bbx_model():
    
#    cnn = resnet50.ResNet50(weights='imagenet', include_top=True)
    
#    cnn = inception_v3.InceptionV3(weights='imagenet',
#                    input_shape = (img_h, img_w, 3),
#                    include_top=False,
#                    pooling='avg')
    
    
    cnn = inception_resnet_v2.InceptionResNetV2(weights='imagenet',
                    input_shape = (img_h, img_w, 3),
                    include_top=False,
                    pooling='avg')
    
    
    for layer in cnn.layers:
        layer.trainable = True
        
    cnn.trainable = True
    x = cnn.output
    x = Dropout(drop_out_ratio)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(drop_out_ratio)(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    Output = Dense(8, activation='linear')(x)
    model = Model(cnn.input, Output)
    
    return model



if Train_Exist:
    
    
    print('Train Exist Model')
        
    the_epochs = exist_vive_epochs
    the_patience = exist_vive_patience
    
    the_train_X, the_valid_X = train_X, valid_X
    the_train_y_exist, the_valid_y_exist = train_y_exist, valid_y_exist
    
    
    
    
    
    exist_model = build_exist_model()
    
    if restore_exist:
        print('Restore Exist Model')
        exist_model.load_weights(restore_exist_path)
        time.sleep(sleep_time)
    
    exist_model.summary()
    opt = Adam(lr = exist_lr)
    exist_model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['acc'])
    
    
    earlystopping = EarlyStopping(monitor = 'val_loss', patience = the_patience, 
                                  verbose = 1, mode = 'min')
    
    checkpoint = ModelCheckpoint(save_exist_path,
                                 verbose = 1,
                                 save_best_only = True,
                                 save_weights_only = True,
                                 monitor = 'val_loss',
                                 mode = 'min')
    
    
    history = exist_model.fit(the_train_X, the_train_y_exist, 
                              epochs = the_epochs, 
                              batch_size = batch_size,
                              validation_data = (the_valid_X, the_valid_y_exist),
                              callbacks=[earlystopping, checkpoint])
    
    
    
    loss_list = list(history.history['loss'])
    val_loss_list = list(history.history['val_loss'])
    
    acc_list = list(history.history['acc'])
    val_acc_list = list(history.history['val_acc'])
    
    
    exist_history = [loss_list, val_loss_list, acc_list, val_acc_list]
    
    with open('hand_exist_history','wb') as fp:
        pickle.dump(exist_history, fp)
    
    with open('hand_exist_history','rb') as fp:
        exist_history = pickle.load(fp)
        
    plt.plot(exist_history[0])
    plt.plot(exist_history[1])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper right')
    #plt.show()
    plt.savefig('hand_exist_model_loss.png')
    plt.clf()
    
    
    plt.plot(exist_history[2])
    plt.plot(exist_history[3])
    plt.title('model acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    #plt.show()
    plt.savefig('hand_exist_model_acc.png')
    plt.clf()
    
    
    try:
        del exist_model
    except:
        print('no exist model')




time.sleep(sleep_time)



if Train_BBX:
    print('Train BBX Model')
    
    the_epochs = bbx_vive_epochs
    the_patience = bbx_vive_patience
    
    the_train_X, the_valid_X = train_X, valid_X
    the_train_y_bbx, the_valid_y_bbx = train_y_bbx, valid_y_bbx
    
    
    
    bbx_model = build_bbx_model()
    
    if restore_bbx:
        print('Restore BBX Model')
        bbx_model.load_weights(restore_bbx_path)
        time.sleep(sleep_time)
    
       
    bbx_model.summary()
    opt = Adam(lr = bbx_lr)
    bbx_model.compile(loss = 'mse', optimizer = opt)
    
    
    earlystopping = EarlyStopping(monitor = 'val_loss', patience = the_patience, 
                                  verbose = 1, mode = 'min')
    
    checkpoint = ModelCheckpoint(save_bbx_path,
                                 verbose = 1,
                                 save_best_only = True,
                                 save_weights_only = True,
                                 monitor = 'val_loss',
                                 mode = 'min')
    
    history = bbx_model.fit(the_train_X, the_train_y_bbx, 
                              epochs = the_epochs, 
                              batch_size = batch_size,
                              validation_data = (the_valid_X, the_valid_y_bbx),
                              callbacks=[earlystopping, checkpoint])
    
    
    
    loss_list = list(history.history['loss'])
    val_loss_list = list(history.history['val_loss'])
    
    
    bbx_history = [loss_list, val_loss_list]
    
    with open('hand_bbx_history','wb') as fp:
        pickle.dump(bbx_history, fp)
    
    with open('hand_bbx_history','rb') as fp:
        bbx_history = pickle.load(fp)
        
    plt.plot(bbx_history[0])
    plt.plot(bbx_history[1])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim((0,1000))
    plt.legend(['train', 'valid'], loc='upper right')
    #plt.show()
    plt.savefig('hand_bbx_model_loss.png')
    plt.clf()
    
    
    try:
        del bbx_model
    except:
        print('no bbx model')

