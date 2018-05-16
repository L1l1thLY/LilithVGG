from judge_vgg import Judge_vgg
from utils.config import ConfigReader, TrainNetConfig
from scipy.misc import imread, imresize
import argparse
import tensorflow as tf
#
# def parse_args():
#     """Parse input arguments."""
#     parser = argparse.ArgumentParser(description='VGG test demo')
#     parser.add_argument('--net', dest='demo_net', help='Network to use [googlenet incption v1]',
#                         default='InceptionV1')
#     parser.add_argument('--im', dest='im_path', help='Path to the image',
#                         default='data/demo/demo.jpg', type=str)
#     parser.add_argument('--model', dest='model', help='Model path', default='./')
#     parser.add_argument('--meta', dest='meta', help='Dataset meta info, class names',
#                         default='./data/datasets/meta.txt', type=str)
#     args = parser.parse_args()
#
#     return args

def test_net():
    config_reader = ConfigReader('./configs/vgg16.yml')
    test_config = TrainNetConfig(config_reader.get_train_config())


    img = imread("./Data/jpg/Face0.jpg", mode='RGB')
    img = imresize(img, [224, 224])  # height, width

    net = Judge_vgg(test_config)
    net.build_model()

    ckpt_path = './logs/train'
    # start a session
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    print('Model checkpoint path: {}'.format(ckpt_path))
    try:
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except FileNotFoundError:
        raise 'Check your pretrained {:s}'.format(ckpt_path)

    prob = sess.run(net.prob, feed_dict={net.x: [img]})


    print('Classification Result:')
    print(prob)

    sess.close()


if __name__ == '__main__':
    test_net()