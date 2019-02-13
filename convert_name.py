import os
import sys
import numpy as np
import glob

img_dir='./data_sync/image_02/data'
pc_dir = './data_sync/velodyne_points/data'
f_rgb = glob.glob(os.path.join(img_dir,'*.png'))
f_rgb.sort()
data_tag = [name.split('/')[-1].split('.')[-2] for name in f_rgb]

img_dest_dir='./testing/image_2' 
pc_dest_dir ='./testing/velodyne'
calib_dest_dir = './testing/calib'

for name in data_tag:
    os.system('mv %s %s' % ( os.path.join(img_dir, name+'.png'), os.path.join(img_dest_dir, '%06d.png' % int(name))))
    os.system('mv %s %s' % ( os.path.join(pc_dir, name+'.bin'), os.path.join(pc_dest_dir, '%06d.bin' % int(name))))
    os.system('cp %s %s' % ( './calibration/000000.txt', os.path.join(calib_dest_dir, '%06d.txt' % int(name))))

