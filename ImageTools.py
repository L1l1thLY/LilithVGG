import skimage
import skimage.io
import skimage.transform
import numpy as np


# load an image and clip it into square
# then resize it to size*size
def load_image_and_center_clip(path, size = 224):
    img = skimage.io.imread(path)
    # img = img / 255.0
    # assert (0 <= img).all() and (img <= 1.0).all()

    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to (size, size)
    resized_img = skimage.transform.resize(crop_img, (size, size))
    return resized_img

if __name__ == '__main__':
    img = load_image_and_center_clip("./TestData/puzzle.jpeg")
    skimage.io.imsave("./TestData/output.jpg", img)