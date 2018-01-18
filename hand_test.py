#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 13:23:08 2018

@author: HSIN
"""

import sys
import os
import json
import scipy
import numpy as np
import argparse
import pickle
import time

import imageio
from skimage import img_as_float
from skimage.draw import line,circle
from skimage.io import imread, imsave, imshow

#import cv2

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

img_h = 224
img_w = 224


drop_out_ratio = 0.1

batch_size = 8

output_dim_exist = 2
output_dim_bbx = 8    

test_output_label_path = 'test_output_label/'
test_output_img_path = 'test_output_img/'


exist_model_path = sys.argv[1]
bbx_model_path = sys.argv[2]


if not os.path.exists(test_output_label_path):
    os.makedirs(test_output_label_path)
if not os.path.exists(test_output_img_path):
    os.makedirs(test_output_img_path)

def inference(exist_model, bbx_model, img_input_path, img_dump_path, lab_dump_path):
    
    image_data = imageio.imread(img_input_path)
    
    scale_h = image_data.shape[0]
    scale_w = image_data.shape[1]
    
    image_data = img_prepro(image_data)
    
    test_X = []
    test_X.append(image_data)
    test_X = np.asarray(test_X)
    
    pred_lab_exist = exist_model.predict(test_X)[0]
    print('Exist:', pred_lab_exist)
    
    L_exist_conf = pred_lab_exist[0]
    R_exist_conf = pred_lab_exist[1]
    
    pred_lab = bbx_model.predict(test_X)[0]
    
    pred_lab[0] = (pred_lab[0] / img_w) * scale_w
    pred_lab[2] = (pred_lab[2] / img_w) * scale_w
    pred_lab[4] = (pred_lab[4] / img_w) * scale_w
    pred_lab[6] = (pred_lab[6] / img_w) * scale_w
    
    pred_lab[1] = (pred_lab[1] / img_h) * scale_h
    pred_lab[3] = (pred_lab[3] / img_h) * scale_h
    pred_lab[5] = (pred_lab[5] / img_h) * scale_h
    pred_lab[7] = (pred_lab[7] / img_h) * scale_h
    
    pred_lab = pred_lab.astype('int') 
    
    print('BBX', pred_lab)
        
    answer_L = list(pred_lab[:4])
    answer_R = list(pred_lab[4:])    
    
    answers = []
    
#    adjust_range = 10
#    
#    
#    answer_L2 = list(answer_L)
#    answer_L2[0] -= adjust_range * np.random.random()
#    answer_L2[1] -= adjust_range * np.random.random()
#    answer_L2[2] += adjust_range * np.random.random()
#    answer_L2[3] += adjust_range * np.random.random()
#    
#    answer_R2 = list(answer_R)
#    answer_R2[0] -= adjust_range * np.random.random()
#    answer_R2[1] -= adjust_range * np.random.random()
#    answer_R2[2] += adjust_range * np.random.random()
#    answer_R2[3] += adjust_range * np.random.random()
#    
#    
#    answer_L3 = list(answer_L)
#    answer_L3[0] += adjust_range * np.random.random()
#    answer_L3[1] += adjust_range * np.random.random()
#    answer_L3[2] -= adjust_range * np.random.random()
#    answer_L3[3] -= adjust_range * np.random.random()
#    
#    answer_R3 = list(answer_R)
#    answer_R3[0] += adjust_range * np.random.random()
#    answer_R3[1] += adjust_range * np.random.random()
#    answer_R3[2] -= adjust_range * np.random.random()
#    answer_R3[3] -= adjust_range * np.random.random()
#    
    
    
    
    if L_exist_conf >= 0.5:
        the_answer_L = answer_L + [0] + [1]
        answers.append(the_answer_L)
#        the_answer_L = answer_L2 + [0] + [0.8]
#        answers.append(the_answer_L)
#        the_answer_L = answer_L3 + [0] + [0.8]
#        answers.append(the_answer_L)
#        
    if R_exist_conf >= 0.5:
        the_answer_R = answer_R + [1] + [1]
        answers.append(the_answer_R)
#        the_answer_R = answer_R2 + [1] + [0.8]
#        answers.append(the_answer_R)
#        the_answer_R = answer_R3 + [1] + [0.8]
#        answers.append(the_answer_R)
    
    data = {}
    data['bbox'] = {}
    
    L_data = pred_lab[:4]
    R_data = pred_lab[4:]
    
    left_hand_data = []
    for j in range(len(L_data)):
        left_hand_data.append(int(str(L_data[j])))
    
    right_hand_data = []
    for j in range(len(R_data)):
        right_hand_data.append(int(str(R_data[j])))
    
    
    if L_exist_conf >= 0.5:
        data['bbox']['L'] = left_hand_data
    if R_exist_conf >= 0.5:
        data['bbox']['R'] = right_hand_data
        
    with open(lab_dump_path, 'w') as f:
        json.dump(data, f)
        
    draw_bbox(img_path = img_input_path, label_path = lab_dump_path, output_path = img_dump_path)
    
    return answers


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def img_prepro(image_data):
#    retval, image_data = cv2.threshold(image_data,110, 255, cv2.cv2.THRESH_TOZERO)
    image_data = scipy.misc.imresize(image_data, [img_h, img_w, 3])
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







exist_model = build_exist_model()
exist_model.load_weights(exist_model_path)

bbx_model = build_bbx_model()
bbx_model.load_weights(bbx_model_path)

img_paths = judger_hand.get_file_names()
f = judger_hand.get_output_file_object()


for i in range(len(img_paths)):
    
    img_path = img_paths[i]
    img_dump_path = os.path.join(test_output_img_path, str(i) + '_bbox.png')
    lab_dump_path = os.path.join(test_output_label_path, str(i) + '_label.json')
    
    answers = inference(exist_model, bbx_model, img_path, img_dump_path, lab_dump_path)
    
    for j in range(len(answers)):
        box = answers[j]
        to_write = '%s %d %d %d %d %d %f\n' % (img_path, box[0], box[1], box[2], box[3], box[4], box[5] )
        print(i, to_write)
        f.write(to_write.encode())
score, err = judger_hand.judge()
if err is not None:
    print (err)
f.close()



try:
    del exist_model
except:
    print('no exist model')


try:
    del bbx_model
except:
    print('no bbx model')
