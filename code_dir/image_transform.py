#Imports
from libtiff import TIFF 
from matplotlib import pyplot as plt
import sys
import numpy as np 
from img_IO import *

#Coordinate functions
# index -> local
def index_to_local(i, j, pixel_size=(1.0, 1.0), origin_local=(0.0, 0.0)):
    try:
        x = origin_local[0] + j * pixel_size[0]
        y = origin_local[1] + i * pixel_size[1]
        return x, y
    except Exception as e:
        print("Error", e)
        return None, None

# local -> index
def local_to_index(x, y, pixel_size=(1.0, 1.0), origin_local=(0.0, 0.0), H=None, W=None):
    try:
        j = int(round((x - origin_local[0]) / pixel_size[0]))
        i = int(round((y - origin_local[1]) / pixel_size[1]))
        if H is not None:
            i = max(0, min(H-1, i))
        if W is not None:
            j = max(0, min(W-1, j))
        return i, j
    except Exception as e:
        print("Error", e)
        return None, None

# local -> world
def local_to_world(x, y, origin_world=(0.0, 0.0)):
    try:
        X = origin_world[0] + x
        Y = origin_world[1] + y
        return X, Y
    except Exception as e:
        print("Error", e)
        return None, None

# world -> local
def world_to_local(X, Y, origin_world=(0.0, 0.0)):
    try:
        x = X - origin_world[0]
        y = Y - origin_world[1]
        return x, y
    except Exception as e:
        print("Error", e)
        return None, None



#Affine transforms
def affine_transform(x, y, matrix=None, translation=(0, 0)):
    try:
        if matrix is None:
            matrix = np.eye(2, dtype=np.float32)
        point = np.array([x, y], dtype=np.float32)
        t = np.array(translation, dtype=np.float32)
        transformed = matrix @ point + t
        return transformed[0], transformed[1]
    except Exception as e:
        print("Error", e)
        return None, None

def compose_affine(matrix1, trans1, matrix2, trans2):
    try:
        new_matrix = matrix1 @ matrix2
        new_translation = matrix1 @ trans2 + trans1
        return new_matrix, new_translation
    except Exception as e:
        print("Error", e)
        return None, None
    

def nearest_neighbor(img, x, y):

    try:
        i = int(round(y))
        j = int(round(x))
        # Clip to bounds
        i = max(0, min(img.shape[0]-1, i))
        j = max(0, min(img.shape[1]-1, j))
        return img[i, j]
    except Exception as e:
        print("Error", e)
        return None

def bilinear_interpolation(img, x, y):
    try:
        i0 = int(np.floor(y))
        i1 = i0 + 1
        j0 = int(np.floor(x))
        j1 = j0 + 1

        # Clip indices
        i0 = max(0, min(img.shape[0]-1, i0))
        i1 = max(0, min(img.shape[0]-1, i1))
        j0 = max(0, min(img.shape[1]-1, j0))
        j1 = max(0, min(img.shape[1]-1, j1))

        # fractional parts
        dy = y - i0
        dx = x - j0

        # pixel values
        p00 = img[i0, j0]
        p01 = img[i0, j1]
        p10 = img[i1, j0]
        p11 = img[i1, j1]

        # bilinear formula
        val = (1-dx)*(1-dy)*p00 + dx*(1-dy)*p01 + (1-dx)*dy*p10 + dx*dy*p11
        return np.clip(val, 0, 255)
    except Exception as e:
        print("Error:", e)
        return None

def transform_image(file_path, matrix=None, translation=(0,0), method='nearest'):
    img = load_tiff(file_path)
    try:
        H, W = img.shape
        out = np.zeros_like(img)
        inv_matrix = np.eye(2)
        if matrix is not None:
            inv_matrix = np.linalg.inv(matrix)

        for i in range(H):
            for j in range(W):
                # Map output pixel to source image
                src_x, src_y = inv_matrix @ (np.array([j,i]) - np.array(translation))
                if method == 'nearest':
                    val = nearest_neighbor(img, src_x, src_y)
                else:
                    val = bilinear_interpolation(img, src_x, src_y)
                out[i,j] = val
        return out
    except Exception as e:
        print("Error:", e)
        return img
#util
def parse_float_args(start_idx, count):
    return [float(sys.argv[start_idx + i]) for i in range(count)]
if __name__ == '__main__':
    cmd = sys.argv[1]



    if cmd in ['--index_to_local', '-il']:
        i, j = map(int, sys.argv[2:4])
        x, y = index_to_local(i, j)
        print(f"Index ({i},{j}) -> Local ({x:.2f},{y:.2f})")

    elif cmd in ['--local_to_index', '-li']:
        x, y = parse_float_args(2, 2)
        H = int(sys.argv[4]) if len(sys.argv) > 4 else None
        W = int(sys.argv[5]) if len(sys.argv) > 5 else None
        i, j = local_to_index(x, y, H=H, W=W)
        print(f"Local ({x:.2f},{y:.2f}) -> Index ({i},{j})")

    elif cmd in ['--apply_affine', '-a']:
        x, y = parse_float_args(2, 2)
        m00, m01, m10, m11 = parse_float_args(4, 4)
        tx, ty = parse_float_args(8, 2) if len(sys.argv) > 9 else (0, 0)
        matrix = np.array([[m00, m01], [m10, m11]], dtype=np.float32)
        X, Y = affine_transform(x, y, matrix, (tx, ty))
        print(f"Point ({x:.2f},{y:.2f}) -> Transformed ({X:.2f},{Y:.2f})")

    elif cmd in ['--local_to_world', '-lw']:
        x, y = parse_float_args(2, 2)
        ox = float(sys.argv[4]) if len(sys.argv) > 4 else 0
        oy = float(sys.argv[5]) if len(sys.argv) > 5 else 0
        X, Y = local_to_world(x, y, origin_world=(ox, oy))
        print(f"Local ({x:.2f},{y:.2f}) -> World ({X:.2f},{Y:.2f})")

    elif cmd in ['--world_to_local', '-wl']:
        X, Y = parse_float_args(2, 2)
        ox = float(sys.argv[4]) if len(sys.argv) > 4 else 0
        oy = float(sys.argv[5]) if len(sys.argv) > 5 else 0
        x, y = world_to_local(X, Y, origin_world=(ox, oy))
        print(f"World ({X:.2f},{Y:.2f}) -> Local ({x:.2f},{y:.2f})")

    elif cmd in ['--transform_image', '-ti']:
        file_path = sys.argv[2]
        method = sys.argv[3] if len(sys.argv) > 3 else 'nearest'
        output_path = "transformed.tiff"  # default

        # output filename if provided
        if '--save' in sys.argv:
            save_idx = sys.argv.index('--save')
            if save_idx + 1 < len(sys.argv):
                output_path = sys.argv[save_idx + 1]
        elif '-o' in sys.argv:
            save_idx = sys.argv.index('-o')
            if save_idx + 1 < len(sys.argv):
                output_path = sys.argv[save_idx + 1]
        if len(sys.argv) > 7:
            m00, m01, m10, m11 = parse_float_args(4, 4)
            tx, ty = parse_float_args(8, 2) if len(sys.argv) > 9 else (0, 0)
            matrix = np.array([[m00, m01], [m10, m11]], dtype=np.float32)
            translation = (tx, ty)
        else:
            matrix = None
            translation = (0, 0)

        out_img = transform_image(file_path, matrix, translation, method)
        save_tiff(out_img, output_path)
        print(f"Image transformed and saved to {output_path}")
