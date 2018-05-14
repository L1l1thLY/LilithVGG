import vgg16
import tensorflow as tf
import image_tools
import score_tools
import numpy as np

class Lilith_vgg(vgg16.Vgg16):
    def __init__(self, true_sex_false_age=True, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            print("Vgg16: Do not find npy file.")
            # TODO: Find a default path properly.

        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()

        print("Vgg16: npy file loaded")
        self.sex_or_age = true_sex_false_age
        self.var_dict = {}

    def build(self, rgb_image):
        if self.sex_or_age is True:
            self.build_sex_vgg16(rgb_image)
        else:
            self.build_age_vgg16(rgb_image)

    def build_sex_vgg16(self, rgb_image, name="vgg_sex"):
        bgr_image = image_tools.convert_rgb_to_bgr_for_vgg(rgb_image)
        assert bgr_image.get_shape().as_list()[1:] == [224, 224, 3]
        print("Sex_vgg_16: data checking finished.")
        print("Sex_vgg_16: building...")
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

            self.fc8 = self.fc_layer_original_vgg(self.relu7, "fc8-2")

            self.prob = tf.nn.softmax(self.fc8, name="prob")

            self.data_dict = None

    def build_age_vgg16(self, rgb_image, name="vgg_age"):
        bgr_image = image_tools.convert_rgb_to_bgr_for_vgg(rgb_image)
        assert bgr_image.get_shape().as_list()[1:] == [224, 224, 3]
        print("Sex_vgg_16: data checking finished.")
        print("Sex_vgg_16: building...")
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

            self.fc8 = self.fc_layer_original_vgg(self.relu7, "fc8-101")

            self.prob = tf.nn.softmax(self.fc8, name="prob")

            self.data_dict = None



    def get_conv_kernel_constant(self, name):
        return tf.constant(self.data_dict[name]['weights'], name="kernel")

    def get_conv_biases_constant(self, name):
        return tf.constant(self.data_dict[name]['biases'], name="biases")

    def get_fc_weight_constant(self, name):
        return tf.constant(self.data_dict[name]['weights'], name="weights")

    def get_fc_biases_constant(self, name):
        return tf.constant(self.data_dict[name]['biases'], name="biases")



def test_gender():
    # Data input configuration
    patch_num = 1

    img1 = image_tools.load_image_and_center_clip("./TestData/33woman.jpg")
    batch1 = img1.reshape((1, 224, 224, 3))

    vgg = Lilith_vgg(true_sex_false_age=True, vgg16_npy_path="./PretrainedData/gender_vgg16.npy")


    with tf.device('/cpu:0'):
        images = tf.placeholder("float", [patch_num, 224, 224, 3])
    # vgg.build_original_vgg16(images)
        vgg.build(images)


    sess_config = tf.ConfigProto(
        log_device_placement=True,
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.7,
            allow_growth=True
        )
    )
    feed_dict = {images: batch1}

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        prob = sess.run(vgg.prob, feed_dict=feed_dict)

    print(prob)

def test_age():
    # Data input configuration
    patch_num = 1

    img1 = image_tools.load_image_and_center_clip("./TestData/33woman.jpg")
    batch1 = img1.reshape((1, 224, 224, 3))

    vgg = Lilith_vgg(true_sex_false_age=False, vgg16_npy_path="./PretrainedData/age_vgg16.npy")

    with tf.device('/cpu:0'):
        images = tf.placeholder("float", [patch_num, 224, 224, 3])
        # vgg.build_original_vgg16(images)
        vgg.build(images)

    sess_config = tf.ConfigProto(
        log_device_placement=True,
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=0.7,
            allow_growth=True
        )
    )
    feed_dict = {images: batch1}

    with tf.Session(config=sess_config) as sess:
        sess.run(tf.global_variables_initializer())
        prob = sess.run(vgg.prob, feed_dict=feed_dict)

    print(prob)

    array = prob[0]
    print(array)

    sum_result = 0
    for i in range(0, 101):
        sum_result = sum_result + i * array[i]

    print(sum_result)


if __name__ == '__main__':
   test_gender()
  # test_age()
