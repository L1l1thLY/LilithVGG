import vgg16
import tensorflow as tf
import image_tools
import score_tools
import numpy as np

class Lilith_vgg(vgg16.Vgg16):
    pass

if __name__ == '__main__':

    # Data input configuration
    patch_num = 2

    img1 = image_tools.load_image_and_center_clip("./TestData/puzzle.jpeg")
    img2 = image_tools.load_image_and_center_clip("./TestData/tiger.jpeg")
    batch1 = img1.reshape((1, 224, 224, 3))
    batch2 = img2.reshape((1, 224, 224, 3))

    batches = np.concatenate((batch1, batch2), 0)

    vgg = Lilith_vgg("./PretrainedData/vgg16.npy")

    images = tf.placeholder("float", [patch_num, 224, 224, 3])
    feed_dict = {images: batches}
    # vgg.build_original_vgg16(images)
    vgg.build_trainable_vgg16(images)


    sess_config = tf.ConfigProto(
        log_device_placement=True,
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.7,
            allow_growth=True)
    )

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        prob = sess.run(vgg.prob, feed_dict=feed_dict)

    print(prob)

    score_tools.print_prob(prob[0], './PretrainedData/synset.txt')
    score_tools.print_prob(prob[1], './PretrainedData/synset.txt')