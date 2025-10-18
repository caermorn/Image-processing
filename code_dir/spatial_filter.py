# Imports
from libtiff import TIFF 
from matplotlib import pyplot as plt
import sys
import numpy as np 
from img_IO import *
from intensity_transform import power_transform_array

#util
def to_float(img_uint8):
    return img_uint8.astype(np.float32) / 255.0

def to_uint8(img_float):
    return normalize_uint8(img_float)

#Classes 
class Image8bit:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.uint8)

    def to_float(self):
        return ImageFloat(to_float(self.data))

class ImageFloat:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)

    def to_uint8(self):
        return Image8bit(to_uint8(self.data))



def convolve2d(image, kernel):
    try:
        i_h, i_w = image.shape
        k_h, k_w = kernel.shape
        pad_h = k_h // 2
        pad_w = k_w // 2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        out = np.zeros_like(image, dtype=np.float32)

        for y in range(i_h):
            for x in range(i_w):
                region = padded[y:y + k_h, x:x + k_w]
                out[y, x] = np.sum(region * kernel)
        return out
    except Exception as e:
        print("Error", e)
        return None

def gaussian_kernel(size=5, sigma=1.0):
    try:
        assert size % 2 == 1
        ax = np.arange(-size // 2 + 1, size // 2 + 1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel
    except Exception as e:
        print("Error", e)
        return None


def sobel():
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32) / 8.0
    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32) / 8.0
    return kx, ky


def gradient_magnitude(img_float):
    kx, ky = sobel()
    gx = convolve2d(img_float, kx)
    gy = convolve2d(img_float, ky)
    mag = np.sqrt(gx**2 + gy**2)
    if mag.max() > 0:
        mag = mag / mag.max()
    return mag


def laplac(img_float):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)
    return convolve2d(img_float, kernel)


def gaussian_blur(img_float, size=5, sigma=1.0):
    kernel = gaussian_kernel(size, sigma)
    return convolve2d(img_float, kernel)

def unsharp_mask(img_float, size=5, sigma=1.0, strength=1.0):
    blurred = gaussian_blur(img_float, size, sigma)
    sharp = img_float + strength * (img_float - blurred)
    sharp = np.clip(sharp, 0, 1)
    return sharp

def laplacian_sharpen(img_float, strength=1.0):
    lap = laplac(img_float)
    sharp = img_float - strength * lap
    sharp = np.clip(sharp, 0, 1)
    return sharp


def process_gaussian(file_path, size=5, sigma=1.0):
    img8_data = load_tiff(file_path)
    if img8_data is None:
        return
    imgf = ImageFloat(to_float(img8_data))
    blurred = gaussian_blur(imgf.data, size, sigma)
    show_grayscale(to_uint8(blurred), f"Gaussian Blur ({size}x{size}, sigma={sigma})")

def process_gradient(file_path):
    img8_data = load_tiff(file_path)
    if img8_data is None:
        return
    imgf = ImageFloat(to_float(img8_data))
    grad = gradient_magnitude(imgf.data)
    show_grayscale(to_uint8(grad), "Gradient Magnitude")

def process_laplacian(file_path):
    img8_data = load_tiff(file_path)
    if img8_data is None:
        return
    imgf = ImageFloat(to_float(img8_data))
    lap = laplac(imgf.data)
    show_grayscale(to_uint8(lap), "Laplacian")

def process_unsharp(file_path, size=5, sigma=1.0, strength=1.0):
    img8_data = load_tiff(file_path)
    if img8_data is None:
        return
    imgf = ImageFloat(to_float(img8_data))
    sharp = unsharp_mask(imgf.data, size, sigma, strength)
    show_grayscale(to_uint8(sharp), f"Unsharp Mask (strength={strength})")

def process_lap_sharpen(file_path, strength=1.0):
    img8_data = load_tiff(file_path)
    if img8_data is None:
        return
    imgf = ImageFloat(to_float(img8_data))
    sharp = laplacian_sharpen(imgf.data, strength)
    show_grayscale(to_uint8(sharp), f"Laplacian Sharpen (strength={strength})")

def example3_43(file_path, lap_strength=1.0, smooth_size=5, smooth_sigma=1.0, gamma=1.0):

    # Step a: load image
    img8_data = load_tiff(file_path)
    if img8_data is None:
        return
    img_float = ImageFloat(to_float(img8_data))
    a = img_float.data

    # Step b: Laplace transform
    b = laplac(a)

    # Step c: Sharpen by a + b
    c = np.clip(a + lap_strength * b, 0, 1)

    # Step d: Sobel gradient of a
    d = gradient_magnitude(a)

    # Step e: Gaussian blur
    e = gaussian_blur(d, size=smooth_size, sigma=smooth_sigma)

    # Step f: Mask from c * e
    f = c * e
    f = np.clip(f, 0, 1)

    # Step g: Sharpen by a + f
    g = np.clip(a + f, 0, 1)

    # Step h: Power-law
    g_uint8 = to_uint8(g)
    h_uint8 = power_transform_array(g_uint8, gamma)

    show_grayscale(h_uint8)


if __name__ == '__main__':
    cmd = sys.argv[1]
    file_path = sys.argv[2]

    if cmd in ['--gaussian', '-g']:
        size = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        sigma = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
        process_gaussian(file_path, size, sigma)

    elif cmd in ['--gradient', '-grad']:
        process_gradient(file_path)

    elif cmd in ['--laplacian', '-lap']:
        process_laplacian(file_path)

    elif cmd in ['--unsharp', '-us']:
        size = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        sigma = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
        strength = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
        process_unsharp(file_path, size, sigma, strength)

    elif cmd in ['--lap-sharpen', '-ls']:
        strength = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        process_lap_sharpen(file_path, strength)

    elif cmd in ['--example3_43', '-ex']:
        # Optional args
        lap_strength = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        smooth_size = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        smooth_sigma = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
        gamma = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0

        example3_43(file_path, lap_strength, smooth_size, smooth_sigma, gamma)