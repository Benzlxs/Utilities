"""
Desciption:
    This file mainly includes the algorithm to split the whole dataset into trainig data and testing data.
    The name are saved in the training.txt and val.txt
"""


import os
import numpy as np
import glob
import random

DIR_label = '/home/b/Kitti/object/training/label_2/'
DIR_files = DIR_label + '*' + '.txt'
all_files = glob.glob(DIR_files)

split_ratio = 0.75  ## split_ratio
num = len(all_files)

traing_txt_dir = 'training.txt'
val_txt_dir = 'val.txt'
train_txt_array = np.array([])
val_txt_array = np.array([])
for i in range(num):
    base_name = os.path.splitext(os.path.basename(all_files[i]))[0]
    base_name = np.array([base_name])
    rand_num = random.random()
    if rand_num<=split_ratio:
        ## saveing in training.txt
        print('{}into training dataset'.format(base_name))
        #np.savetxt(traing_txt_dir, base_name, newline='\n', fmt="%s")
        train_txt_array = np.append(train_txt_array,base_name)
    else:
        ## saving in val.txt
        #np.savetxt(val_txt_dir, base_name, newline='\n', fmt='%s')
        print('{}into val dataset'.format(base_name))
        val_txt_array = np.append(val_txt_array, base_name)

np.savetxt(traing_txt_dir, train_txt_array, newline='\n', fmt="%s")
np.savetxt(val_txt_dir, val_txt_array, newline='\n', fmt="%s")
