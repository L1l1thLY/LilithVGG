import base_network
import tensorflow as tf


class Judge_vgg(base_network.Net):
    def __init__(self, cfg_):
        super().__init__(cfg_)
        self.x = tf.placeholder(tf.float32, name='x', shape=[self.config.batch_size,
                                                             self.config.image_width,
                                                             self.config.image_height,
                                                             self.config.image_depth], )
        self.y = tf.placeholder(tf.int16, name='y', shape=[self.config.batch_size,
                                                           self.config.n_classes])
        self.loss = None
        self.accuracy = None
        self.summary = []

    def get_summary(self):
            return self.summary

    def conv2d(self, layer_name, inputs, out_channels, kernel_size=3, strides=1, padding='SAME'):
        in_channels = inputs.get_shape()[-1]
        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope
            w = tf.get_variable(name='weights',
                                trainable=True,
                                shape=[kernel_size, kernel_size, in_channels, out_channels],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='biases',
                                trainable=True,
                                shape=[out_channels],
                                initializer=tf.constant_initializer(0.0))
            inputs = tf.nn.conv2d(inputs, w, [1, strides, strides, 1], padding=padding, name='conv')
            inputs = tf.nn.bias_add(inputs, b, name='bias_add')
            inputs = tf.nn.relu(inputs, name='relu')
            return inputs

    def max_pool(self, layer_name, inputs, pool_size=2, strides=2, padding='SAME'):
        with tf.name_scope(layer_name):
            return tf.nn.max_pool(inputs, [1, pool_size, pool_size, 1], [1, strides, strides, 1], padding=padding,
                                  name=layer_name)

    def avg_pool(self, layer_name, inputs, pool_size, strides, padding='SAME'):
        with tf.name_scope(layer_name):
            return tf.nn.avg_pool(inputs, [1, pool_size, pool_size, 1], [1, strides, strides, 1], padding=padding,
                                  name=layer_name)

    def dropout(self, layer_name, inputs, keep_prob):
        # dropout_rate = 1 - keep_prob
        with tf.name_scope(layer_name):
            return tf.nn.dropout(name=layer_name, x=inputs, keep_prob=keep_prob)

    def fc(self, layer_name, inputs, out_nodes):
        shape = inputs.get_shape()
        if len(shape) == 4:  # x is 4D tensor
            size = shape[1].value * shape[2].value * shape[3].value
        else:  # x has already flattened
            size = shape[-1].value
        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope
            w = tf.get_variable('weights',
                                shape=[size, out_nodes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('biases',
                                shape=[out_nodes],
                                initializer=tf.constant_initializer(0.0))
            flat_x = tf.reshape(inputs, [-1, size])
            inputs = tf.nn.bias_add(tf.matmul(flat_x, w), b)
            inputs = tf.nn.relu(inputs)
            return inputs

    def cal_loss(self, logits, labels):
        with tf.name_scope('loss') as scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name='cross-entropy')
            self.loss = tf.reduce_mean(cross_entropy, name='loss')
            loss_summary = tf.summary.scalar(scope, self.loss)
            self.summary.append(loss_summary)

    def cal_accuracy(self, logits, labels):
        with tf.name_scope('accuracy') as scope:
            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            correct = tf.cast(correct, tf.float32)
            self.accuracy = tf.reduce_mean(correct) * 100.0
            accuracy_summary = tf.summary.scalar(scope, self.accuracy)
            self.summary.append(accuracy_summary)

    def optimize(self):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
            # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(self.loss, global_step=self.global_step_tensor)
            return train_op

#def conv2d(self, layer_name, inputs, out_channels, kernel_size, strides=1, padding='SAME')
    def build_model(self):
        conv1_1 = self.conv2d("conv1_1", self.x, 64)
        conv1_2 = self.conv2d("conv1_2", conv1_1, 64)
        pool1 = self.max_pool("pool1", conv1_2)

        conv2_1 = self.conv2d("conv2_1", pool1, 128)
        conv2_2 = self.conv2d("conv2_2", conv2_1, 128)
        pool2 = self.max_pool("pool2", conv2_2)

        conv3_1 = self.conv2d("conv3_1", pool2, 256)
        conv3_2 = self.conv2d("conv3_2", conv3_1, 256)
        conv3_3 = self.conv2d("conv3_3", conv3_2, 256)
        pool3 = self.max_pool("pool3", conv3_3)

        conv4_1 = self.conv2d("conv4_1", pool3, 512)
        conv4_2 = self.conv2d("conv4_2", conv4_1, 512)
        conv4_3 = self.conv2d("conv4_3", conv4_2, 512)
        pool4 = self.max_pool("pool4", conv4_3)

        conv5_1 = self.conv2d("conv5_1", pool4, 512)
        conv5_2 = self.conv2d("conv5_2", conv5_1, 512)
        conv5_3 = self.conv2d("conv5_3", conv5_2, 512)
        pool5 = self.max_pool("pool5", conv5_3)

        fc6 = self.fc("fc6", pool5, 4096)
        fc7 = self.fc("fc7", fc6, 4096)
        fc8_2 = self.fc("fc8_2", fc7, out_nodes=self.config.n_classes)

        self.cal_loss(fc8_2, self.y)
        self.cal_accuracy(fc8_2, self.y)
        train_op = self.optimize()
        return train_op