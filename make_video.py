###
import numpy as np
import os
import sys
import glob
from moviepy.editor import ImageSequenceClip


file_list = glob.glob(os.path.join('./image_02/','*.png'))
file_list.sort()

clip = ImageSequenceClip(file_list, fps=15)
name = '15.gif'
#clip.write_gif(name, fps=15)
clip.write_videofile("15.mp4",fps=15)

