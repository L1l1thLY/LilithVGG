import vgg16
import tensorflow as tf
import image_tools
import score_tools
import numpy as np

class Lilith_vgg(vgg16.Vgg16):
    pass

if __name__ == '__main__':

    # Load data
    image1 = image_tools.load_image_and_center_clip("./TestData/tiger.jpeg")
    image1_true_result = [1 if i == 292 else 0 for i in range(1000)]  # 1-hot result for tiger

    batch = image1.reshape((1, 224, 224, 3))

    # Data input configuration
    patch_num = 1
    vgg = Lilith_vgg("./PretrainedData/vgg16.npy")
    true_out = tf.placeholder(tf.float32, [1, 1000])
    images = tf.placeholder("float", [patch_num, 224, 224, 3])

    # Sorry to my poor gpu
    with tf.device('/cpu:0'):
        vgg.build_trainable_vgg16(images)
        cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
        train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

    sess_config = tf.ConfigProto(
        log_device_placement=True,
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.9,
            allow_growth=True
        )
    )

    feed_dict = {images: batch}

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train, feed_dict={images: batch, true_out: [image1_true_result]})
        prob = sess.run(vgg.prob, feed_dict={images: batch})

    print(prob)

    score_tools.print_prob(prob[0], './PretrainedData/synset.txt')
