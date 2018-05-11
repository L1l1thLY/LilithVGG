import skimage
import skimage.io
import skimage.transform
import tensorflow as tf

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
    resized_img = skimage.transform.resize(crop_img, (size, size), preserve_range=True)
    return resized_img

# convert rgb to bgr image and sub the vgg mean
def convert_rgb_to_bgr_for_vgg(rgb_image_tensor):
    r_tensor, g_tensor, b_tensor = tf.split(rgb_image_tensor, 3, 3)
    r_tensor, g_tensor, b_tensor = sub_image_vgg_mean(r_tensor, g_tensor, b_tensor)
    bgr_image_tensor = tf.concat([b_tensor, g_tensor, r_tensor], 3)
    return bgr_image_tensor

# sub the vgg mean
def sub_image_vgg_mean(r_tensor, g_tensor, b_tensor):
    vgg_mean = [103.939, 116.779, 123.68]
    # BGR MEAN
    new_r = r_tensor - vgg_mean[2]
    new_g = g_tensor - vgg_mean[1]
    new_b = b_tensor - vgg_mean[0]
    return new_r, new_g, new_b

if __name__ == '__main__':
    img = load_image_and_center_clip("./TestData/puzzle.jpeg")
    skimage.io.imsave("./TestData/output.jpg", img)
