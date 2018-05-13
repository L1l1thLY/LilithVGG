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
        self.var_dict = {}

    def build_original_vgg16(self, rgb_image, name="vgg_net"):
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

    # Build a tranable vgg 16
    # whose layers can be initialized by
    # new value or pretrain value.
    def build_trainable_vgg16(self, rgb_image, name="trainable_vgg_net"):
        bgr_image = image_tools.convert_rgb_to_bgr_for_vgg(rgb_image)
        assert bgr_image.get_shape().as_list()[1:] == [224, 224, 3]
        print("Vgg16: data checking finished.")
        print("Vgg16: building...")

        with tf.name_scope(name):
            self.conv1_1 = self.conv_layer_trainable(bgr_image, 3, 64, "conv1_1")
            self.conv1_2 = self.conv_layer_trainable(self.conv1_1, 64, 64, "conv1_2")
            self.pool1 = self.max_pool(self.conv1_2, 'pool1')

            self.conv2_1 = self.conv_layer_trainable(self.pool1, 64, 128, "conv2_1")
            self.conv2_2 = self.conv_layer_trainable(self.conv2_1, 128, 128, "conv2_2")
            self.pool2 = self.max_pool(self.conv2_2, 'pool2')

            self.conv3_1 = self.conv_layer_trainable(self.pool2, 128, 256, "conv3_1")
            self.conv3_2 = self.conv_layer_trainable(self.conv3_1, 256, 256, "conv3_2")
            self.conv3_3 = self.conv_layer_trainable(self.conv3_2, 256, 256, "conv3_3")
            self.pool3 = self.max_pool(self.conv3_3, "pool3")

            self.conv4_1 = self.conv_layer_trainable(self.pool3, 256, 512, "conv4_1")
            self.conv4_2 = self.conv_layer_trainable(self.conv4_1, 512, 512, "conv4_2")
            self.conv4_3 = self.conv_layer_trainable(self.conv4_2, 512, 512, "conv4_3")
            self.pool4 = self.max_pool(self.conv4_3, "pool4")

            self.conv5_1 = self.conv_layer_trainable(self.pool4, 512, 512, "conv5_1")
            self.conv5_2 = self.conv_layer_trainable(self.conv5_1, 512, 512, "conv5_2")
            self.conv5_3 = self.conv_layer_trainable(self.conv5_2, 512, 512, "conv5_3")
            self.pool5 = self.max_pool(self.conv5_3, "pool5")

            self.fc6 = self.fc_layer_trainable(self.pool5, 4096, "fc6")
            assert self.fc6.get_shape().as_list()[1:] == [4096]
            self.relu6 = tf.nn.relu(self.fc6)

            self.fc7 = self.fc_layer_trainable(self.relu6, 4096, "fc7")
            self.relu7 = tf.nn.relu(self.fc7)

            self.fc8 = self.fc_layer_trainable(self.relu7, 1000, "fc8")

            self.prob = tf.nn.softmax(self.fc8, name="prob")

            self.data_dict = None


    def max_pool(self, input_data, name):
        return tf.nn.max_pool(input_data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, input_data, name):
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

        return relu

    def conv_layer_trainable(self, input_data, in_channels_num, out_channels_num, name, new_value=False):
        kernel, biases = self.get_conv_val(name, 3, in_channels_num, out_channels_num, new_value=new_value)
        with tf.name_scope(name):
            conv = tf.nn.conv2d(
                input_data,
                kernel,
                [1, 1, 1, 1],
                padding="SAME",
                name=name
            )
            bias = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(bias)

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


    def fc_layer_trainable(self, input_data, output_data_size, name, new_value=False):
        with tf.name_scope(name):
            input_shape = input_data.get_shape().as_list()
            dim = 1
            for d in input_shape[1:]:
                dim *= d
            x = tf.reshape(input_data, [-1, dim])

            weights, biases = self.get_fc_val(name, dim, output_data_size, new_value=new_value)

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_kernel_constant(self, name):
        return tf.constant(self.data_dict[name][0], name="kernel")

    def get_conv_biases_constant(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight_constant(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

    def get_fc_biases_constant(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def update_fc_var_dict(self, name, weights_var, biases_var):
        self.var_dict[(name, 0)] = weights_var
        self.var_dict[(name, 1)] = biases_var

    def update_conv_var_dict(self, name, kernel_var, biases_var):
        self.var_dict[(name, 0)] = kernel_var
        self.var_dict[(name, 1)] = biases_var


    def get_conv_val(self, name, kernel_size, in_channels_num, out_channels_num, new_value=False):
        with tf.variable_scope(name):
            if new_value == True:
                kernel = tf.Variable(
                    tf.truncated_normal([kernel_size, kernel_size, in_channels_num, out_channels_num], 0.0, 0.001),
                    name="new_kernel"
                )
                biases = tf.Variable(
                    tf.truncated_normal([out_channels_num], 0.0, 0.001),
                    name="new_biases"
                )
            else:
                kernel = tf.Variable(
                    self.get_conv_kernel_constant(name),
                    name="pretrained_kernel"
                )
                biases = tf.Variable(
                    self.get_conv_biases_constant(name),
                    name="pretrained_biases"
                )
        self.update_conv_var_dict(name, kernel, biases)
        return kernel, biases

    def get_fc_val(self, name, in_data_size, out_data_size, new_value=False):
        with tf.variable_scope(name):
            if new_value == True:
                weights = tf.Variable(
                    tf.truncated_normal([in_data_size, out_data_size], 0.0, 0.001),
                    name="new_weight"
                )
                biases = tf.Variable(
                    tf.truncated_normal([out_data_size], 0.0, 0.001),
                    name="new_biases"
                )
            else:
                weights = tf.Variable(
                    self.get_fc_weight_constant(name),
                    name="pretrained_weights"
                )
                biases = tf.Variable(
                    self.get_fc_biases_constant(name),
                    name="pretrained_biases"
                )
        self.update_fc_var_dict(name, weights, biases)
        return weights, biases

    def save_var_as_npy(self, sess, path="./vgg16-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in self.var_dict.items():
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(path, data_dict)
        print("file saved", path)
        return path


if __name__ == '__main__':

    # Data input configuration
    patch_num = 2

    img1 = image_tools.load_image_and_center_clip("./TestData/puzzle.jpeg")
    img2 = image_tools.load_image_and_center_clip("./TestData/tiger.jpeg")
    batch1 = img1.reshape((1, 224, 224, 3))
    batch2 = img2.reshape((1, 224, 224, 3))

    batches = np.concatenate((batch1, batch2), 0)

    vgg = Vgg16("./vgg16-save.npy")

    images = tf.placeholder("float", [patch_num, 224, 224, 3])
    feed_dict = {images: batches}
    # vgg.build_original_vgg16(images)
    vgg.build_trainable_vgg16(images)


    sess_config = tf.ConfigProto(
        log_device_placement=True,
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.7,
            allow_growth=True
        )
    )

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        prob = sess.run(vgg.prob, feed_dict=feed_dict)
        vgg.save_var_as_npy(sess)
    print(prob)

    score_tools.print_prob(prob[0], './PretrainedData/synset.txt')
    score_tools.print_prob(prob[1], './PretrainedData/synset.txt')
