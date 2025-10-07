#Imports
from libtiff import TIFF 
from matplotlib import pyplot as plt
import sys
import numpy as np 
from img_IO import *

#Power law 

def power_transform(file_path, gamma):
    try:
        imgs = load_tiff(file_path)
        if not imgs:
            return None
        img = normalize_uint8(imgs[0])
        img_float = img / 255.0
        img_trans = np.clip(img_float ** gamma, 0, 1)
        return (img_trans * 255).astype(np.uint8)
    except Exception as e:
        print("Error:", e)
        return None

    

if __name__ == '__main__':
    cmd = sys.argv[1] 

    if cmd in ['--power', '-p']:
        gamma = float(sys.argv[3])
        img_trans = power_transform(sys.argv[2], gamma)
        if img_trans is not None:
            show_grayscale(img_trans)
            