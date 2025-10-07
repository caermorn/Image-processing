# Library imports
from libtiff import TIFF 
from matplotlib import pyplot as plt
import sys
import numpy as np 


# Normalize to uint8 
def normalize_uint8(img):

    min_val = img.min()
    max_val = img.max()
    if max_val == min_val:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# Load a .tiff or .svs
def load_tiff(file_path: str):
    try:
        tiff = TIFF.open(file_path, mode='r')
        first_page = next(tiff.iter_images())
        tiff.close()
        return [first_page]
    except Exception as e:
        print("Error loading file:", e)
        return []


# Display functions
def show_grayscale(img, title=None):
    img_uint8 = normalize_uint8(img) 
    plt.imshow(img_uint8, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


def show_color(img, title=None):
    img_uint8 = normalize_uint8(img) if img.dtype != np.uint8 else img  
    plt.imshow(img_uint8, interpolation='nearest', vmin=0, vmax=255)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Load a subregion from the first page
def load_subregion(path, x, y, w, h):
    try:
        imgs = load_tiff(path)
        if not imgs:
            return None
        img = imgs[0]
        H, W = img.shape[:2]
        # Clip coordinates
        x1, y1 = max(0, min(W, x + w)), max(0, min(H, y + h))
        x, y = max(0, x), max(0, y)
        sub = img[y:y1, x:x1]
        return sub
    except Exception as e:
        print("Error loading subregion:", e)
        return None

# Combine grayscale channels into RGB
def combine_to_rgb(red_path, green_path, blue_path, out_path):
    r_imgs = load_tiff(red_path)
    g_imgs = load_tiff(green_path)
    b_imgs = load_tiff(blue_path)
    
    if not r_imgs or not g_imgs or not b_imgs:
        print("Could not load one of the input files")
        return

    r, g, b = r_imgs[0], g_imgs[0], b_imgs[0]  # Only first page

    if r.shape != g.shape or r.shape != b.shape:
        print("input sizes do not match")
        return

    rgb = np.stack([normalize_uint8(r),
                    normalize_uint8(g),
                    normalize_uint8(b)], axis=-1)
    show_color(rgb, title='Combined RGB Image')

    try:
        t = TIFF.open(out_path, 'w')
        t.write_image(rgb)
        t.close()
        print(f"Saved combined RGB to {out_path}")
    except Exception as e:
        print("Error saving image:", e)

# Main logic
if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == '--show-full':
        imgs = load_tiff(sys.argv[2])
        for i, img in enumerate(imgs):
            if img.ndim == 2:
                show_grayscale(img, f"Page {i}")
            else:
                show_color(img, f"Page {i}")

    elif cmd == '--show-subregion':
        if len(sys.argv) != 7:
            sys.exit(0)
        x, y, w, h = map(int, sys.argv[3:7])
        sub = load_subregion(sys.argv[2], x, y, w, h)
        if sub is not None:
            if sub.ndim == 2:
                show_grayscale(sub, f"Subregion {x},{y} {w}x{h}")
            else:
                show_color(sub, f"Subregion {x},{y} {w}x{h}")

    elif cmd == '--combine-rgb':
        if len(sys.argv) != 6:
            sys.exit(0)
        combine_to_rgb(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
