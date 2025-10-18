#Imports
from libtiff import TIFF 
from matplotlib import pyplot as plt
import sys
import numpy as np 
from img_IO import *

#Power law 

def power_transform(file_path, gamma):
    try:
        img = load_tiff(file_path)
        if img is None:
            return None
        img_float = img / 255.0
        img_trans = np.clip(img_float ** gamma, 0, 1)
        return (img_trans * 255).astype(np.uint8)

    except Exception as e:
        print("Error:", e)
        return None
    
def power_transform_LUT(file_path, gamma):
    try:
        transform_func = lambda p: 255 * ((p / 255) ** gamma)
        LUT = build_LUT(transform_func)
        img_trans = apply_LUT(file_path, LUT)
        return img_trans
    except Exception as e:
        print("Error:", e)
        return None

#Piecewise linear
def piecewise_linear(file_path, r1, s1, r2, s2):
    try:
        img = load_tiff(file_path)
        if img is None:
            return None

        def transform_func(p):
            if p < r1:
                return (s1 / r1) * p
            elif p < r2:
                return ((s2 - s1) / (r2 - r1)) * (p - r1) + s1
            else:
                return ((255 - s2) / (255 - r2)) * (p - r2) + s2

        # Vectorized
        vectorized_transform = np.vectorize(transform_func)
        img_trans = vectorized_transform(img)
        img_trans = np.clip(img_trans, 0, 255).astype(np.uint8)

        return img_trans
    except Exception as e:
        print("Error:", e)
        return None
    
def piecewise_linear_LUT(file_path, r1, s1, r2, s2):
    try:
        def transform_func(p):
            if p < r1:
                return (s1 / r1) * p
            elif p < r2:
                return ((s2 - s1) / (r2 - r1)) * (p - r1) + s1
            else:
                return ((255 - s2) / (255 - r2)) * (p - r2) + s2

        LUT = build_LUT(transform_func)
        img_trans = apply_LUT(file_path, LUT)
        return img_trans
    except Exception as e:
        print("Error:", e)
        return None

#Lookup table
def build_LUT(transform_func):
    try:
        LUT = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            LUT[i] = np.clip(transform_func(i), 0, 255)
        return LUT
    except Exception as e:
        print("Error:", e)
        return None


def apply_LUT(file_path, LUT):
    try:
        img = load_tiff(file_path)
        if img is None:
            return None
        img_trans = LUT[img]
        return img_trans
    except Exception as e:
        print("Error:", e)
        return None


#Histogram calculation
def calc_hist(file_path):
    try:
        img = load_tiff(file_path)
        if img is None:
            return None

        hist = np.bincount(img.flatten(), minlength=256)
        return hist
    except Exception as e:
        print("Error:", e)
        return None

def equal_hist(file_path):
    try:
        img = load_tiff(file_path)
        hist = np.bincount(img.flatten(), minlength=256)
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]

        # Map original pixels to equalized values
        img_eq = np.interp(img.flatten(), range(256), cdf_normalized * 255).astype(np.uint8)

        return img_eq.reshape(img.shape)
    except Exception as e:
        print("Error:", e)
        return None

#Show histogram 
def show_hist(file_path, title="Histogram"):
    hist = calc_hist(file_path)

    plt.figure(figsize=(8, 4))
    plt.bar(range(256), hist, width=1.0, color='black')
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.xlim([0, 255])
    plt.show()

if __name__ == '__main__':
    cmd = sys.argv[1] 
    file_path = sys.argv[2]
    if cmd in ['--power', '-p']:
        gamma = float(sys.argv[3])
        img_trans = power_transform(file_path, gamma)
        show_grayscale(img_trans)
            
    elif cmd in ['--histogram', '--hist']:
        show_hist(file_path)

    elif cmd in ['--hist_equalize', '-he']:
        img_trans = equal_hist(file_path)
        show_grayscale(img_trans)

    elif cmd in ['--piecewise', '-pw']:
        r1 = int(sys.argv[3])
        s1 = int(sys.argv[4])
        r2 = int(sys.argv[5])
        s2 = int(sys.argv[6])
        img_trans = piecewise_linear(file_path, r1, s1, r2, s2)
        show_grayscale(img_trans)

    elif cmd in ['--power_lut', '-pl']:
        gamma = float(sys.argv[3])
        img_trans = power_transform_LUT(file_path, gamma)
        show_grayscale(img_trans)

    elif cmd in ['--piecewise_lut', '-pwl']:
        r1 = int(sys.argv[3])
        s1 = int(sys.argv[4])
        r2 = int(sys.argv[5])
        s2 = int(sys.argv[6])
        img_trans = piecewise_linear_LUT(file_path, r1, s1, r2, s2)
        show_grayscale(img_trans)
