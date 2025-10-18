import sys
import numpy as np
import matplotlib.pyplot as plt
from libtiff import TIFF

# Normalize image to uint8
def normalize_uint8(img):
    min_val = img.min()
    max_val = img.max()
    if max_val == min_val:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# Load a .tiff 
def load_tiff(file_path: str):
    try:
        tiff = TIFF.open(file_path, mode='r')
        first_img = next(tiff.iter_images(), None)  # Take only the first page
        tiff.close()

        if first_img is None:
            print("Error: No images found in TIFF.")
            return None

        return normalize_uint8(first_img)

    except Exception as e:
        print("Error loading file:", e)
        return None

# save .tiff 
def save_tiff(img, file_path):
    try:
        if img.dtype != np.uint8:
            img = normalize_uint8(img)

        tiff = TIFF.open(file_path, mode='w')
        tiff.write_image(img)
        tiff.close()
        print(f"Saved image to {file_path}")
    except Exception as e:
        print("Error:", e)

# Display functions
def show_grayscale(img, title=None):
    plt.imshow(img, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def show_color(img, title=None):
    plt.imshow(img, interpolation='nearest', vmin=0, vmax=255)
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

        # Clip coordinates properly
        x1 = min(W, x + w)
        y1 = min(H, y + h)
        x0 = max(0, x)
        y0 = max(0, y)

        sub = img[y0:y1, x0:x1]
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
        print("Input sizes do not match")
        return

    rgb = np.stack([r, g, b], axis=-1) 
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
        print('Wrong call!')
    cmd = sys.argv[1]
    file_path = sys.argv[2]

    if cmd in ['--show', '-s']:
        imgs = load_tiff(sys.argv[2])
        for i, img in enumerate(imgs):
            if img.ndim == 2:
                show_grayscale(img, f"Page {i}")
            else:
                show_color(img, f"Page {i}")

    elif cmd in ['--subregion', '-sr']:
        if len(sys.argv) != 7:
            sys.exit(0)
        x, y, w, h = map(int, sys.argv[3:7])
        sub = load_subregion(sys.argv[2], x, y, w, h)
        if sub is not None:
            if sub.ndim == 2:
                show_grayscale(sub, f"Subregion {x},{y} {w}x{h}")
            else:
                show_color(sub, f"Subregion {x},{y} {w}x{h}")

    elif cmd in ['--combine-rgb']:
        if len(sys.argv) != 6:
            sys.exit(0)
        combine_to_rgb(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
