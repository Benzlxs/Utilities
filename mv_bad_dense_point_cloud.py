import os
import sys
import numpy as np
import glob
from pathlib import Path


dense_path = Path('./dense_point_cloud')
old_dense_path= Path('./bad_dense_point_cloud')


with open('bad_fles.txt', 'r') as f:
    fs = f.readlines()

filenames = [f.rstrip() for f in fs]

for name in filenames:
    target_file = dense_path/name
    os.system('mv %s %s'%(target_file, old_dense_path))

