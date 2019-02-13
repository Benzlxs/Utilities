# algorithm steps
# step1: get all names of new labels
# step2: renew the name
# step3: move image, labels, velody, calib, planes with new name into training
# dataset
# step4: put the name into train.txt, trainval.txt and val.txt

import os
import sys
import numpy as np
import glob

label_dir = '../testing/labels'

all_files = glob.glob(os.path.join(label_dir,'*.txt'))
all_files.sort()

num_files = len(all_files)

source_file = './test.txt'

with open(source_file,'w') as f:
    for i in range(num_files):
        name_old = []
        name_old = os.path.basename(all_files[i]).split('.')
        name_new = '01'+name_old[0][2:]
        print(name_new)
        ## copying the image
        if os.path.exists(os.path.join('../testing/image_2', name_old[0]+'.png')):
            os.system('cp %s %s'% (os.path.join('../testing/image_2', name_old[0]+'.png'),\
                               os.path.join('../training/image_2', name_new + '.png')))
        else:
            raise Exception('image %s doest not exist'% (name_old[0]+'.png'))
        ## copying the calib file
        if os.path.exists(os.path.join('../testing/calib', name_old[0]+'.txt')):
            os.system('cp %s %s'% (os.path.join('../testing/calib', name_old[0]+'.txt'),\
                               os.path.join('../training/calib', name_new + '.txt')))
        else:
             raise Exception('calib %s doest not exist'% (name_old[0]+'.txt'))

        ## copying the point cloud
        if os.path.exists(os.path.join('../testing/velodyne', name_old[0]+'.bin')):
            os.system('cp %s %s'% (os.path.join('../testing/velodyne', name_old[0]+'.bin'),\
                               os.path.join('../training/velodyne', name_new + '.bin')))
        else:
             raise Exception('point cloud %s doest not exist'% (name_old[0]+'.bin'))

        ## copying the planes
        if os.path.exists(os.path.join('../testing/planes', name_old[0]+'.txt')):
            os.system('cp %s %s'% (os.path.join('../testing/planes', name_old[0]+'.txt'),\
                               os.path.join('../training/planes', name_new + '.txt')))
        else:
             raise Exception('planes %s doest not exist'% (name_old[0]+'.txt'))

        ## copying the lables
        if os.path.exists(os.path.join('../testing/labels', name_old[0]+'.txt')):
            os.system('cp %s %s'% (os.path.join('../testing/labels', name_old[0]+'.txt'),\
                               os.path.join('../training/label_2', name_new + '.txt')))
        else:
             raise Exception('labels %s doest not exist'% (name_old[0]+'.txt'))

        f.write('{}\n'.format(name_new))

#if os.path.exists(os.path.join(sph5_dir,line[0:6] +'.sph5')):
 #    os.system('rm %s' % os.path.join(sph5_dir,line[0:6]+'.sph5'))
