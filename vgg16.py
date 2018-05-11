import tensorflow as tf
import os
import numpy as np
import image_tools
import score_tools

class Vgg16:
    def __init__(self, vgg16_npy_path = None):
        if vgg16_npy_path is None:
            print("Vgg16: Do not find npy file.")
            # TODO: Find a default path properly.

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("Vgg16: npy file loaded")

    def build(self, rgb_image, name = "vgg_net", train_mode=None):
        bgr_image = image_tools.convert_rgb_to_bgr_for_vgg(rgb_image)
        assert bgr_image.get_shape().as_list()[1:] == [224, 224, 3]
        print("Vgg16: data checking finished.")
        print("Vgg16: building...")

        with tf.name_scope(name):
            self.conv1_1 = self.conv_layer(bgr_image, "conv1_1")
            self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
            self.pool1 = self.max_pool(self.conv1_2, 'pool1')

            self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
            self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
            self.pool2 = self.max_pool(self.conv2_2, 'pool2')

            self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
            self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
            self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
            self.pool3 = self.max_pool(self.conv3_3, 'pool3')

            self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
            self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
            self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
            self.pool4 = self.max_pool(self.conv4_3, 'pool4')

            self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
            self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
            self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
            self.pool5 = self.max_pool(self.conv5_3, 'pool5')

            self.fc6 = self.fc_layer_original_vgg(self.pool5, "fc6")

            assert self.fc6.get_shape().as_list()[1:] == [4096]
            self.relu6 = tf.nn.relu(self.fc6)

            self.fc7 = self.fc_layer_original_vgg(self.relu6, "fc7")
            self.relu7 = tf.nn.relu(self.fc7)

            self.fc8 = self.fc_layer_original_vgg(self.relu7, "fc8")

            self.prob = tf.nn.softmax(self.fc8, name="prob")

            self.data_dict = None

    def max_pool(self, input_data, name):
        return tf.nn.max_pool(input_data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, input_data, name, log=False):
        with tf.variable_scope(name):
            kernel = self.get_conv_kernel_constant(name)
            biases = self.get_conv_biases_constant(name)
            conv = tf.nn.conv2d(
                input=input_data,
                filter=kernel,
                strides=[1, 1, 1, 1],
                padding='SAME',
                name=name
            )
            bias = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(bias)
            if log is True:
                print(conv.name)
            return relu

    def fc_layer_original_vgg(self, input_data, name, log=False):
        with tf.variable_scope(name):
            input_shape = input_data.get_shape().as_list()
            dim = 1
            for d in input_shape[1:]:
                dim *= d
            x = tf.reshape(input_data, [-1, dim])

            weights = self.get_fc_weight_constant(name)
            biases = self.get_fc_biases_constant(name)

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            if log is True:
                print("VGG16: fc_layer_original_vgg input_shape", input_shape)
            return fc


    def get_conv_kernel_constant(self, name):
        return tf.constant(self.data_dict[name][0], name="kernel")

    def get_conv_biases_constant(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight_constant(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

    def get_fc_biases_constant(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")



if __name__ == '__main__':

    # Data input configuration
    patch_num = 2

    img1 = image_tools.load_image_and_center_clip("./TestData/puzzle.jpeg")
    img2 = image_tools.load_image_and_center_clip("./TestData/tiger.jpeg")
    batch1 = img1.reshape((1, 224, 224, 3))
    batch2 = img2.reshape((1, 224, 224, 3))

    batches = np.concatenate((batch1, batch2), 0)

    vgg = Vgg16("./PretrainedData/vgg16.npy")
    sess_config = tf.ConfigProto(
        log_device_placement=True,
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.7,
            allow_growth=True)
    )
    with tf.Session(config=sess_config) as sess:
        images = tf.placeholder("float", [patch_num, 224, 224, 3])
        feed_dict = {images: batches}
        vgg.build(images)
        prob = sess.run(vgg.prob, feed_dict=feed_dict)
        print(prob)

        score_tools.print_prob(prob[0], './PretrainedData/synset.txt')
        score_tools.print_prob(prob[1], './PretrainedData/synset.txt')